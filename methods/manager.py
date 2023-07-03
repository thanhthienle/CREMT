from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader

from .swag import SWAG
from .model import *
from .backbone import *
from .prompt import *
from .utils import *
from methods.multitask.weight_methods import NashMTL, METHODS

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import copy
import random
import numpy as np
from itertools import cycle

from sklearn.mixture import GaussianMixture

from tqdm import tqdm, trange


class Manager(object):
    def __init__(self, args):
        super().__init__()
        if args.mtl is not None:
            self.train_classifier = self._train_mtl_classifier
        else:
            self.train_classifier = self._train_normal_classifier

    def _train_mtl_classifier(self, args, classifier, swag_classifier, replayed_epochs, nash_mtl_object=None, name=""):
        assert nash_mtl_object is not None, "Really bro?"

        classifier.train()
        swag_classifier.train()

        modules = [classifier]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        modules_params_list = list(modules_parameters)

        optimizer = torch.optim.Adam(
            [
                dict(params=modules_parameters, lr=args.classifier_lr),
                dict(params=nash_mtl_object.parameters(), lr=args.mtl_lr),
            ],
        )

        def train_data(data_loader_, name=name):
            past_losses, cur_losses = [], []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            past_sampled = 0
            cur_sampled = 0

            total_hits = 0
            total_past_hits = 0
            total_cur_hits = 0
            
            for (past_labels, past_tokens, _), (cur_labels, cur_tokens, _) in td:
                optimizer.zero_grad()

                # batching
                past_sampled += len(past_labels)
                past_targets = past_labels.type(torch.LongTensor).to(args.device)
                past_tokens = torch.stack([x.to(args.device) for x in past_tokens], dim=0)

                cur_sampled += len(cur_labels)
                cur_targets = cur_labels.type(torch.LongTensor).to(args.device)
                cur_tokens = torch.stack([x.to(args.device) for x in cur_tokens], dim=0)

                sampled += past_sampled + cur_sampled

                # classifier forward
                past_reps = classifier(past_tokens)
                cur_reps = classifier(cur_tokens)

                # prediction
                past_probs = F.softmax(past_reps, dim=1)
                _, past_pred = past_probs.max(1)
                past_hits = (past_pred == past_targets).float()   

                cur_probs = F.softmax(cur_reps, dim=1)
                _, cur_pred = cur_probs.max(1)
                cur_hits = (cur_pred == cur_targets).float()

                # accuracy
                total_past_hits += past_hits.sum().data.cpu().numpy().item()
                total_cur_hits += cur_hits.sum().data.cpu().numpy().item()
                total_hits += total_past_hits + total_cur_hits

                # loss components
                past_loss = F.cross_entropy(input=past_reps, target=past_targets, reduction="mean")
                cur_loss = F.cross_entropy(input=cur_reps, target=cur_targets, reduction="mean")

                objectives = torch.stack(
                    (past_loss, cur_loss,)
                )
                
                past_losses.append(past_loss.item())
                cur_losses.append(cur_loss.item())

                _, _ = nash_mtl_object.backward(
                    losses=objectives,
                    shared_parameters=modules_params_list
                )

                # params update
                # torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(
                    past_loss = np.array(past_losses).mean(),
                    cur_loss = np.array(cur_losses).mean(),
                    cur_acc = total_cur_hits / cur_sampled,
                    ovr_acc = total_hits / sampled,
                )

        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            combined_data_loader = zip(
                get_data_loader(args, [instance for instance in replay_data if instance["relation"] not in self.cur_rel_ids], shuffle=True),
                cycle(get_data_loader(args, [instance for instance in replay_data if instance["relation"] in self.cur_rel_ids], shuffle=True)),
            )
            train_data(combined_data_loader, f"{name}{e_id + 1}")

            # SWAG
            data_loader = get_data_loader(args, replay_data, shuffle=True)
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

    def _train_normal_classifier(self, args, classifier, swag_classifier, replayed_epochs, nash_mtl_object=None, name=""):
        classifier.train()
        swag_classifier.train()

        modules = [classifier]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.classifier_lr}])

        def train_data(data_loader_, name=name):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0
            for step, (labels, tokens, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # classifier forward
                reps = classifier(tokens)

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                hits = (pred == targets).float()

                # accuracy
                total_hits += hits.sum().data.cpu().numpy().item()

                # loss components
                loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                losses.append(loss.item())
                loss.backward()

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)

        for e_id in range(args.classifier_epochs):
            data_loader = get_data_loader(args, replayed_epochs[e_id % args.replay_epochs], shuffle=True)
            train_data(data_loader, f"{name}{e_id + 1}")
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

    def train_embeddings(self, args, encoder, training_data, task_id):
        encoder.train()
        classifier = Classifier(args=args).to(args.device)
        classifier.train()
        data_loader = get_data_loader(args, training_data, shuffle=True)

        modules = [classifier, encoder.encoder.embeddings]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.encoder_lr}])

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0

            for step, (labels, tokens, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # encoder forward
                encoder_out = encoder(tokens)

                # classifier forward
                reps = classifier(encoder_out["x_encoded"])

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # loss components
                CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                loss = CE_loss
                losses.append(loss.item())
                loss.backward()

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)

        for e_id in range(args.encoder_epochs):
            train_data(data_loader, f"train_embeddings_epoch_{e_id + 1}", e_id)

    def train_prompt_pool(self, args, encoder, prompt_pool, classifier, training_data, task_id):
        encoder.eval()
        classifier.train()
        modules = [classifier, prompt_pool]
        if task_id == 0 and args.encoder_seprarate_embeddings == True:
            print("Separating Embeddings")
            modules.append(encoder.encoder.embeddings)
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.prompt_pool_lr}])

        # get new training data (label, tokens, key) for prompt pool training
        data_loader = get_data_loader(args, training_data, shuffle=True)
        new_training_data = []
        td = tqdm(data_loader, desc=f"get_prompt_key_task_{task_id+1}")
        for step, (labels, tokens, _) in enumerate(td):
            # batching
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            # encoder forward
            encoder_out = encoder(tokens)

            tokens = tokens.cpu().detach().numpy()
            x_key = encoder_out["x_encoded"].cpu().detach().numpy()
            # add to new training data
            for i in range(len(labels)):
                new_training_data.append({"relation": labels[i], "tokens": tokens[i], "key": x_key[i]})
            td.set_postfix()

        # new data loader
        data_loader = get_data_loader(args, new_training_data, shuffle=True)

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0

            replay_sampled = 0
            replay_total_hits = 0

            for step, (labels, tokens, keys, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                x_key = torch.stack([x.to(args.device) for x in keys], dim=0)

                # encoder forward
                encoder_out = encoder(tokens, prompt_pool, x_key)

                # classifier forward
                reps = classifier(encoder_out["x_encoded"])

                # loss components
                prompt_reduce_sim_loss = -args.pull_constraint_coeff * encoder_out["reduce_sim"]

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # replay
                if task_id != 0 and args.replay_ratio != 0:
                    try:
                        (replay_labels, replay_tokens, _) = next(replay_iterator)
                    except:
                        replay_size = max(1, round(args.batch_size * args.replay_ratio))
                        replay_iterator = iter(get_data_loader(args, random.choice(self.replayed_data), shuffle=True, batch_size=replay_size))
                        (replay_labels, replay_tokens, _) = next(replay_iterator)

                    # batching
                    replay_sampled += len(replay_labels)
                    replay_targets = replay_labels.type(torch.LongTensor).to(args.device)
                    replay_tokens = torch.stack([x.to(args.device) for x in replay_tokens], dim=0)

                    # classifier forward
                    replay_reps = classifier(replay_tokens)

                    # prediction
                    replay_probs = F.softmax(replay_reps, dim=1)
                    _, replay_pred = replay_probs.max(1)
                    replay_total_hits += (replay_pred == replay_targets).float().sum().data.cpu().numpy().item()

                    CE_loss = F.cross_entropy(input=torch.cat([reps, replay_reps], dim=0), target=torch.cat([targets, replay_targets], dim=0), reduction="mean")
                else:
                    CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean")

                # losses
                loss = CE_loss + prompt_reduce_sim_loss
                losses.append(loss.item())
                loss.backward()

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled, replay=f"{replay_total_hits}/{replay_sampled}")

        for e_id in range(args.prompt_pool_epochs):
            train_data(data_loader, f"train_prompt_pool_epoch_{e_id + 1}", e_id)

    @torch.no_grad()
    def sample_memorized_data(self, args, encoder, prompt_pool, relation_data, name, task_id):
        encoder.eval()
        data_loader = get_data_loader(args, relation_data, shuffle=False)
        td = tqdm(data_loader, desc=name)

        # output dict
        out = {}

        # x_data
        x_key = []
        x_encoded = []

        for step, (labels, tokens, _) in enumerate(td):
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            x_key.append(encoder(tokens)["x_encoded"])
            x_encoded.append(encoder(tokens, prompt_pool, x_key[-1])["x_encoded"])

        x_key = torch.cat(x_key, dim=0)
        x_encoded = torch.cat(x_encoded, dim=0)

        key_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_key.cpu().detach().numpy())
        encoded_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_encoded.cpu().detach().numpy())

        if args.gmm_num_components == 1:
            key_mixture.weights_[0] = 1.0
            encoded_mixture.weights_[0] = 1.0

        out["replay_key"] = key_mixture
        out["replay"] = encoded_mixture
        return out

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, classifier, prompted_classifier, test_data, name, task_id):
        # models evaluation mode
        encoder.eval()
        classifier.eval()

        # data loader for test set
        data_loader = get_data_loader(args, test_data, shuffle=False)

        # tqdm
        td = tqdm(data_loader, desc=name)

        # initialization
        sampled = 0
        total_hits = np.zeros(4)

        # testing
        for step, (labels, tokens, _) in enumerate(td):
            sampled += len(labels)
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

            # encoder forward
            encoder_out = encoder(tokens)

            # prediction
            reps = classifier(encoder_out["x_encoded"])
            probs = F.softmax(reps, dim=1)
            _, pred = probs.max(1)

            # accuracy_0
            total_hits[0] += (pred == targets).float().sum().data.cpu().numpy().item()

            # pool_ids
            pool_ids = [self.id2taskid[int(x)] for x in pred]
            for i, pool_id in enumerate(pool_ids):
                total_hits[1] += pool_id == self.id2taskid[int(labels[i])]

            # get pools
            prompt_pools = [self.prompt_pools[x] for x in pool_ids]

            # prompted encoder forward
            prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

            # prediction
            reps = prompted_classifier(prompted_encoder_out["x_encoded"])
            probs = F.softmax(reps, dim=1)
            _, pred = probs.max(1)

            # accuracy_2
            total_hits[2] += (pred == targets).float().sum().data.cpu().numpy().item()

            # pool_ids
            pool_ids = [self.id2taskid[int(x)] for x in labels]

            # get pools
            prompt_pools = [self.prompt_pools[x] for x in pool_ids]

            # prompted encoder forward
            prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

            # prediction
            reps = prompted_classifier(prompted_encoder_out["x_encoded"])
            probs = F.softmax(reps, dim=1)
            _, pred = probs.max(1)

            # accuracy_3
            total_hits[3] += (pred == targets).float().sum().data.cpu().numpy().item()

            # display
            td.set_postfix(acc=np.round(total_hits / sampled, 3))
        return total_hits / sampled

    def train(self, args):
        for seed_id in range(args.total_rounds):
            # initialize test results list
            test_cur = []
            test_total = []

            # replayed data
            self.replayed_key = [[] for e_id in range(args.replay_epochs)]
            self.replayed_data = [[] for e_id in range(args.replay_epochs)]

            # random seed
            random.seed(args.seed + seed_id*100)

            # sampler
            sampler = data_sampler(args=args, seed=args.seed + seed_id*100)
            self.rel2id = sampler.rel2id
            self.id2rel = sampler.id2rel

            # convert
            self.id2taskid = {}

            # model
            encoder = BertRelationEncoder(config=args).to(args.device)

            # pools
            self.prompt_pools = []

            # nash_mtl object
            if args.mtl is not None:
                if args.mtl != "nashmtl":
                    raise NotImplementedError()
                else: nash_mtl = METHODS[args.mtl](n_tasks=2, device=args.device, max_norm=args.max_grad_norm)
            else: nash_mtl = None

            # initialize memory
            self.memorized_samples = {}

            # load data and start computation
            all_train_tasks = []
            all_tasks = []
            seen_data = {}

            # task predictor
            task_predictor = Classifier(args=args).to(args.device)
            swag_task_predictor = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

            # classifier
            prompted_classifier = Classifier(args=args).to(args.device)
            swag_prompted_classifier = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                writer = open(f"live_{args.logname}", "a")
                print("=" * 100)
                print(f"task={steps+1}")
                print(f"current relations={current_relations}")

                # Live results
                writer.write("=" * 100)
                writer.write("\n")
                writer.write(f"task={steps+1}\n")
                writer.write(f"current relations={current_relations}\n")

                self.steps = steps
                self.not_seen_rel_ids = [rel_id for rel_id in range(args.num_tasks * args.rel_per_task) if rel_id not in [self.rel2id[relation] for relation in seen_relations]]

                # initialize
                cur_training_data = []
                cur_test_data = []

                # Current relations ids
                self.cur_rel_ids = []

                for i, relation in enumerate(current_relations):
                    cur_training_data += training_data[relation]
                    seen_data[relation] = training_data[relation]
                    cur_test_data += test_data[relation]

                    rel_id = self.rel2id[relation]
                    self.cur_rel_ids.append(rel_id)
                    self.id2taskid[rel_id] = steps
                self.cur_rel_ids = tuple(self.cur_rel_ids)

                # train encoder
                if steps == 0:
                    self.train_embeddings(args, encoder, cur_training_data, task_id=steps)
                    encoder.encoder.first_task_embeddings = copy.deepcopy(encoder.encoder.embeddings)
                    encoder.encoder.first_task_embeddings.eval()

                # new prompt pool
                self.prompt_pools.append(Prompt(args).to(args.device))
                self.train_prompt_pool(args, encoder, self.prompt_pools[-1], prompted_classifier, cur_training_data, task_id=steps)

                # memory
                for i, relation in enumerate(current_relations):
                    self.memorized_samples[sampler.rel2id[relation]] = self.sample_memorized_data(args, encoder, self.prompt_pools[steps], training_data[relation], f"sampling_relation_{i+1}={relation}", steps)

                # replay data for classifier
                for relation in current_relations:
                    print(f"replaying data {relation}")
                    rel_id = self.rel2id[relation]
                    replay_data = self.memorized_samples[rel_id]["replay"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                    for e_id in range(args.replay_epochs):
                        for x_encoded in replay_data[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                            self.replayed_data[e_id].append({"relation": rel_id, "tokens": x_encoded})

                for relation in current_relations:
                    print(f"replaying key {relation}")
                    rel_id = self.rel2id[relation]
                    replay_key = self.memorized_samples[rel_id]["replay_key"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                    for e_id in range(args.replay_epochs):
                        for x_encoded in replay_key[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                            self.replayed_key[e_id].append({"relation": rel_id, "tokens": x_encoded})

                # all
                all_train_tasks.append(cur_training_data)
                all_tasks.append(cur_test_data)

                # train
                self._train_normal_classifier(args, task_predictor, swag_task_predictor, self.replayed_key, None, "train_task_predictor_epoch_")
                if steps == 0:
                    self._train_normal_classifier(args, prompted_classifier, swag_prompted_classifier, self.replayed_data, None, "train_prompted_classifier_epoch_")
                else:
                    self.train_classifier(args, prompted_classifier, swag_prompted_classifier, self.replayed_data, nash_mtl, "train_prompted_classifier_epoch_")

                # prediction
                print("===NON-SWAG===")
                writer.write("===NON-SWAG===\n")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([len(i_th_test_data), self.evaluate_strict_model(args, encoder, task_predictor, prompted_classifier, i_th_test_data, f"test_task_{i+1}", steps)])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                writer.write(f"current test accuracy: {cur_acc}\n")
                writer.write(f"history test accuracy: {total_acc}\n")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                print("===SWAG===")
                writer.write("===SWAG===\n")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([len(i_th_test_data), self.evaluate_strict_model(args, encoder, swag_task_predictor, swag_prompted_classifier, i_th_test_data, f"test_task_{i+1}", steps)])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                writer.write(f"current test accuracy: {cur_acc}\n")
                writer.write(f"history test accuracy: {total_acc}\n")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                print("===UNTIL-NOW==")
                writer.write("===UNTIL-NOW==\n")
                print("accuracies:")
                writer.write("accuracies:\n")
                for x in test_cur:
                    print(x)
                    writer.write(x + "\n")
                print("arverages:")
                for x in test_total:
                    print(x)
                    writer.write(x + "\n")
                writer.close()
