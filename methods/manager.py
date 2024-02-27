from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader

from .swag import SWAG
from .model import *
from .backbone import *
from .prompt import *
from .utils import *
from methods.multitask.grad_mtl import METHODS as GRAD_METHODS
from methods.multitask.weight_methods import NashMTL, PCGrad, METHODS as WEIGHT_METHODS

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from vae import *

import copy
import random
import numpy as npå

from tqdm import tqdm, trange


@torch.no_grad()
def convert_data_tokens_to_queries(args, data, encoder):
    data_loader = get_data_loader(args, data, shuffle=False)
    queries = []
    print("Forward data...")
    for (_, tokens, _) in tqdm(data_loader):
        tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
        queries.append(encoder(tokens))
    queries = torch.cat(queries, dim=0).cpu()

    new_data = copy.deepcopy(data)
    for i in range(len(new_data)):
        new_data[i]["tokens"] = queries[i]
    return new_data


def etf_logitize(args, past_targets):
    num_elements = past_targets.shape[0]
    num_columns = args.rel_per_task * args.num_tasks
    new_tensor = torch.ones(num_elements, num_columns, device=args.device) / (1 - num_columns)
    new_tensor[range(num_elements), past_targets] = 1
    return new_tensor


class Manager(object):
    def __init__(self, args):
        super().__init__()
        if args.mtl is not None:
            try:
                self.train_classifier = getattr(self, f"_train_mtl_classifier_{args.tasktype}")
                if args.tasktype == "ratio":
                    print("INITIALIZED PAST ALPHAS")
                    self.past_alphas = [1.0,]
            except:
                raise NotImplementedError("This task type is not defined.")
        else:
            self.train_classifier = self._train_normal_classifier

        try:
            self.sample_memorized_data = getattr(self, f"sample_{gen_abvr[args.generative]}_data")
        except:
            raise NotImplementedError()

    def _train_classifier_naive(self, args, encoder, classifier, current_task_data, name=""):
        encoder.eval()
        classifier.train()
        optimizer = torch.optim.Adam([dict(params=classifier.parameters(), lr=args.classifier_lr),])

        def train_data(data_loader_, name=name):
            losses = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            hits = 0

            for labels, queries, _ in td:
                optimizer.zero_grad()
                classifier.zero_grad()

                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                queries = torch.stack([x.to(args.device) for x in queries], dim=0)

                # train
                logits = classifier(queries)
                loss = F.cross_entropy(input=logits, target=targets, reduction="mean")
                losses.append(loss.item())
                loss.backward()
                
                # prediction
                _, pred = logits.max(1)
                hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # params update
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                classifier.zero_grad()
                
                # display
                td.set_postfix(
                    loss = np.array(losses).mean(),
                    acc = hits / sampled,
                )
        
        # Train
        current_data_loader = get_data_loader(args, current_task_data, shuffle=True)
        for e_id in range(args.classifier_epochs):
            train_data(current_data_loader, f"{name}{e_id + 1}")
        return classifier

    def _train_mtl_classifier_distill(self, args, encoder, classifier, past_classifier, swag_classifier, replayed_epochs, current_task_data, name=""):
        encoder.eval()
        classifier.train()
        overfit_classifier = copy.deepcopy(classifier)
        overfit_classifier.train()
        overfit_classifier = self._train_classifier_naive(args, encoder, overfit_classifier, current_task_data, name="train_naive_classifier_epoch_")
        overfit_classifier.eval()
        past_classifier.eval()
        
        optimizer = torch.optim.Adam([dict(params=classifier.parameters(), lr=args.classifier_lr),])

        def train_data(data_loader_, name=name):
            distill_past_losses, distill_overfit_losses, past_losses, losses = [], [], [], []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            past_sampled = 0
            cur_sampled = 0

            total_hits = 0
            total_past_hits = 0
            total_cur_hits = 0

            for past_batch, current_gauss_batch in td:
                optimizer.zero_grad()

                past_labels, past_queries, _ = past_batch
                cur_labels, cur_tokens, _ = current_gauss_batch

                # get past_distill_targets
                past_sampled += len(past_labels)
                past_targets = past_labels.type(torch.LongTensor).to(args.device)
                past_queries = torch.stack([x.to(args.device) for x in past_queries], dim=0)
                with torch.no_grad():
                    past_distill_targets = F.softmax(past_classifier(past_queries), dim=1, dtype=torch.float32)

                # get overfit_targets
                cur_sampled += len(cur_labels)
                cur_targets = cur_labels.type(torch.LongTensor).to(args.device)
                cur_queries = torch.stack([x.to(args.device) for x in cur_tokens], dim=0)
                with torch.no_grad():
                    overfit_targets = F.softmax(overfit_classifier(cur_queries), dim=1, dtype=torch.float32)

                sampled += past_sampled + cur_sampled

                # classifier forward
                cur_reps = classifier(cur_queries)
                past_reps = classifier(past_queries)

                # loss components
                distill_loss_past = F.cross_entropy(input=past_reps, target=past_distill_targets, reduction="mean")
                distill_loss_overfit = F.cross_entropy(input=cur_reps, target=overfit_targets, reduction="mean")
                past_loss = F.cross_entropy(input=past_reps, target=past_targets, reduction="mean")
                loss = F.cross_entropy(input=cur_reps, target=cur_targets, reduction="mean")
                
                # Backward and optimize
                distill_loss_past.backward(retain_graph=True)
                distill_past_grad = []
                for param in classifier.parameters():
                    distill_past_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                distill_past_grad = torch.cat(distill_past_grad, dim=0)

                distill_loss_overfit.backward(retain_graph=True)
                distill_overfit_grad = []
                for param in classifier.parameters():
                    distill_overfit_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                distill_overfit_grad = torch.cat(distill_overfit_grad, dim=0)

                past_loss.backward()
                past_grad = []
                for param in classifier.parameters():
                    past_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                past_grad = torch.cat(past_grad, dim=0)

                loss.backward()
                current_grad = []
                for param in classifier.parameters():
                    current_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                current_grad = torch.cat(current_grad, dim=0)

                shared_grad = GRAD_METHODS[args.mtl](torch.stack(
                    [distill_past_grad, distill_overfit_grad, past_grad, current_grad]),
                    args.c
                )["updating_grad"]

                total_length = 0
                for param in classifier.parameters():
                    length = param.numel()
                    param.grad.data = shared_grad[
                        total_length : total_length + length
                    ].reshape(param.shape)
                    total_length += length

                distill_past_losses.append(distill_loss_past.item())
                distill_overfit_losses.append(distill_loss_overfit.item())
                past_losses.append(past_loss.item())
                losses.append(loss.item())

                # prediction
                _, past_pred = past_reps.max(1)
                past_hits = (past_pred == past_targets).float().sum().data.cpu().numpy().item()

                _, cur_pred = cur_reps.max(1)
                cur_hits = (cur_pred == cur_targets).float().sum().data.cpu().numpy().item()

                # accuracy
                total_past_hits += past_hits
                total_cur_hits += cur_hits
                total_hits += total_past_hits + total_cur_hits

                # params update
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                classifier.zero_grad()

                # display
                td.set_postfix(
                    distill_past_loss = np.array(distill_past_losses).mean(),
                    distill_overfit_loss = np.array(distill_overfit_losses).mean(),
                    past_loss = np.array(past_losses).mean(),
                    loss = np.array(losses).mean(),
                    past_acc = total_past_hits / past_sampled,
                    cur_acc = total_cur_hits / cur_sampled,
                    ovr_acc = total_hits / sampled,
                )

        current_relids = self.relids_of_task[-1]
        oversampling_size = args.replay_s_e_e * (len(self.relids_of_task) - 1)

        # consecutive_satisfaction = 0
        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            past_data = []
            current_gauss_data = []
            for instance in replay_data:
                if instance["relation"] in current_relids:
                    current_gauss_data.append(instance)
                else:
                    past_data.append(instance)
            combined_data_loader = zip(
                get_data_loader(args, past_data, shuffle=True),
                get_data_loader(args, random.choices(current_gauss_data, k=oversampling_size), shuffle=True),
            )
            train_data(combined_data_loader, f"{name}{e_id + 1}")

            # SWAG
            # all_data = past_data
            # all_data.extend(current_task_data)
            # data_loader = get_data_loader(args, all_data, shuffle=True)
            # swag_classifier.collect_model(classifier)
            # if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
            #     swag_classifier.sample(0.0)
            #     bn_update(data_loader, swag_classifier)

            # Valid (every 5 epochs) and early stop
            # if (e_id + 1) % 5:
            #     continue

            # _ = self._validation(args, classifier, valid_data=validation_data)
            # if valid_acc >= 0.90:
            #     consecutive_satisfaction += 1
            # else: consecutive_satisfaction = 0

            # if consecutive_satisfaction > 5:
            #     print("EARLY STOP!!!")
            #     break
        
        return classifier

    def _train_mtl_classifier_oldnew(self, args, encoder, classifier, past_classifier, swag_classifier, replayed_epochs, current_task_data, name=""):
        encoder.eval()
        classifier.train()
        swag_classifier.train()

        if args.mtl in ("pcgrad", "nashmtl"):
            param_list = list(classifier.parameters())
            weight_method = WEIGHT_METHODS[args.mtl](n_tasks=2, device=args.device)
            optimizer = torch.optim.Adam(
                [
                    dict(params=classifier.parameters(), lr=args.classifier_lr),
                    dict(params=weight_method.parameters(), lr=args.mtl_lr),
                ],
            )
        else:
            optimizer = torch.optim.Adam([dict(params=classifier.parameters(), lr=args.classifier_lr),])

        def train_data(data_loader_, name=name):
            past_losses, cur_losses = [], []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            past_sampled = 0
            cur_sampled = 0

            total_hits = 0
            total_past_hits = 0
            total_cur_hits = 0

            for past_batch, current_batch in td:
                optimizer.zero_grad()
                classifier.zero_grad()

                past_labels, past_queries, _ = past_batch
                cur_labels, cur_tokens, _ = current_batch

                # batching
                past_sampled += len(past_labels)
                past_targets = past_labels.type(torch.LongTensor).to(args.device)
                past_queries = torch.stack([x.to(args.device) for x in past_queries], dim=0)

                cur_sampled += len(cur_labels)
                cur_targets = cur_labels.type(torch.LongTensor).to(args.device)
                cur_queries = torch.stack([x.to(args.device) for x in cur_tokens], dim=0)

                sampled += past_sampled + cur_sampled

                # classifier forward
                cur_reps = classifier(cur_queries)
                past_reps = classifier(past_queries)
                
                # loss components
                past_loss = F.cross_entropy(input=past_reps, target=past_targets, reduction="mean")
                cur_loss = F.cross_entropy(input=cur_reps, target=cur_targets, reduction="mean")

                if args.mtl in ("pcgrad", "nashmtl"):
                    # Use weight_methods.py
                    losses = torch.stack(
                        (past_loss, cur_loss)
                    )
                    _, _ = weight_method.backward(
                        losses = losses,
                        shared_parameters=param_list,
                        task_specific_parameters=None,
                    )
                else:
                    # Use grad_mtl.py
                    past_loss.backward()
                    past_shared_grad = []
                    for param in classifier.parameters():
                        past_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    past_shared_grad = torch.cat(past_shared_grad, dim=0)

                    cur_loss.backward()
                    cur_shared_grad = []
                    for param in classifier.parameters():
                        cur_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    cur_shared_grad = torch.cat(cur_shared_grad, dim=0)

                    shared_grad = GRAD_METHODS[args.mtl](torch.stack([past_shared_grad, cur_shared_grad]), args.c)["updating_grad"]
                    total_length = 0
                    for param in classifier.parameters():
                        length = param.numel()
                        param.grad.data = shared_grad[
                            total_length : total_length + length
                        ].reshape(param.shape)
                        total_length += length

                # for logging purpose
                past_losses.append(past_loss.item())
                cur_losses.append(cur_loss.item())

                # params update
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                classifier.zero_grad()

                # prediction
                past_probs = F.softmax(past_reps, dim=1)
                _, past_pred = past_probs.max(1)
                past_hits = (past_pred == past_targets).float().sum().data.cpu().numpy().item()

                cur_probs = F.softmax(cur_reps, dim=1)
                _, cur_pred = cur_probs.max(1)
                cur_hits = (cur_pred == cur_targets).float().sum().data.cpu().numpy().item()

                # accuracy
                total_past_hits += past_hits
                total_cur_hits += cur_hits
                total_hits += total_past_hits + total_cur_hits

                # display
                td.set_postfix(
                    past_loss = np.array(past_losses).mean(),
                    cur_loss = np.array(cur_losses).mean(),
                    cur_acc = total_cur_hits / cur_sampled,
                    ovr_acc = total_hits / sampled,
                )

        past_relids = [relid for sublist in self.relids_of_task[:-1] for relid in sublist]
        current_relids = self.relids_of_task[-1]
        past_num_samples = args.replay_s_e_e * (len(self.relids_of_task) - 1)

        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            past_data = []
            for rel_id in past_relids:
                past_data.extend([instance for instance in replay_data if instance["relation"] == rel_id])
            past_data_loader = get_data_loader(args, past_data, shuffle=True)
            current_data_loader = get_data_loader(args, random.choices([instance for instance in replay_data if instance["relation"] in current_relids], k=past_num_samples), shuffle=True)
            combined_data_loader = zip(past_data_loader, current_data_loader)
            train_data(combined_data_loader, f"{name}{e_id + 1}")

    def _train_mtl_classifier_ntask(self, args, encoder, classifier, past_classifier, swag_classifier, replayed_epochs, current_task_data, name=""):
        encoder.eval()
        classifier.train()
        swag_classifier.train()

        optimizer = torch.optim.Adam([dict(params=classifier.parameters(), lr=args.classifier_lr),])

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

            for data_tuple in td:
                optimizer.zero_grad()
                classifier.zero_grad()
                all_shared_grad = []
                last_task_id = len(data_tuple) - 1
                for task_id, (labels, tokens, _) in enumerate(data_tuple):
                    tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                    targets = labels.type(torch.LongTensor).to(args.device)
                    sampled += len(labels)

                    # classifier forward and loss
                    reps = classifier(tokens)
                    task_loss = F.cross_entropy(reps, targets, reduction="mean")
                    task_loss.backward()
                    task_shared_grad = []
                    for param in classifier.parameters():
                        task_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    task_shared_grad = torch.cat(task_shared_grad, dim=0)
                    all_shared_grad.append(task_shared_grad)

                    detached_loss = task_loss.detach()

                    # prediction
                    _, pred = reps.detach().max(dim=1)
                    hits = (pred == targets.detach()).float().sum().data.cpu().numpy().item()
                    total_hits += hits

                    if task_id == last_task_id:
                        cur_losses.append(detached_loss.item())
                        cur_sampled += len(labels)
                        total_cur_hits += hits
                        with open("debug", "a") as writer:
                            writer.write(str(pred))
                            writer.write("\n")
                            writer.write(str(targets.detach()))
                            writer.write("\n\n")
                    else:
                        past_losses.append(detached_loss.item())
                        past_sampled += len(labels)
                        total_past_hits += hits

                shared_grad = GRAD_METHODS[args.mtl](torch.stack(all_shared_grad), args.c)["updating_grad"]
                total_length = 0
                for param in classifier.parameters():
                    length = param.numel()
                    param.grad.data = shared_grad[
                        total_length : total_length + length
                    ].reshape(param.shape)
                    total_length += length

                # params update
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                classifier.zero_grad()
                td.set_postfix(
                    past_loss = np.array(past_losses).mean(),
                    cur_loss = np.array(cur_losses).mean(),
                    cur_acc = total_cur_hits / cur_sampled,
                    ovr_acc = total_hits / sampled,
                )

        past_rel_ids = self.relids_of_task[:-1]
        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            combined_data_loader = zip(
                *[get_data_loader(args, [instance for instance in replay_data if instance["relation"] in task_relids], shuffle=True) for task_relids in past_rel_ids],
                get_data_loader(args, current_task_data, shuffle=True)
            )
            train_data(combined_data_loader, f"{name}{e_id + 1}")

            # SWAG
            past_data = [instance for instance in replay_data if instance["relation"] in flatten_list(past_rel_ids)]
            past_data.extend(current_task_data)
            data_loader = get_data_loader(args, past_data, shuffle=True)
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

    def _train_mtl_classifier_ratio(self, args, encoder, classifier, past_classifier, swag_classifier, replayed_epochs, current_task_data, name=""):
        encoder.eval()
        classifier.train()
        swag_classifier.train()

        optimizer = torch.optim.Adam([dict(params=classifier.parameters(), lr=args.classifier_lr),])

        def train_data(data_loader_, epoch, name=name):
            past_losses, cur_losses = [], []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            past_sampled = 0
            cur_sampled = 0

            total_hits = 0
            total_past_hits = 0
            total_cur_hits = 0

            last_task_id = len(self.relids_of_task) - 1
            local_alpha_old, local_alpha_current = 1., 1.
            for data_tuple in td:
                optimizer.zero_grad()
                classifier.zero_grad()
                old_loss = 0.0
                current_loss = 0.0
                for task_id, (labels, tokens, _) in enumerate(data_tuple):
                    tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                    targets = labels.type(torch.LongTensor).to(args.device)
                    sampled += len(labels)

                    # classifier forward and loss
                    reps = classifier(tokens)

                    # prediction
                    _, pred = reps.detach().max(dim=1)
                    hits = (pred == targets.detach()).float().sum().data.cpu().numpy().item()
                    total_hits += hits

                    if task_id == last_task_id:
                        current_loss = F.cross_entropy(reps, targets, reduction="mean")
                        cur_losses.append(current_loss.detach().item())
                        cur_sampled += len(labels)
                        total_cur_hits += hits

                    else:
                        task_loss = F.cross_entropy(reps, targets, reduction="mean")
                        old_loss += self.past_alphas[task_id] * task_loss
                        past_losses.append(task_loss.detach().item())
                        past_sampled += len(labels)
                        total_past_hits += hits

                old_loss.backward()
                old_shared_grad = []
                for param in classifier.parameters():
                    old_shared_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                old_shared_grad = torch.cat(old_shared_grad, dim=0)

                current_loss.backward()
                current_shared_grad = []
                for param in classifier.parameters():
                    current_shared_grad.append(param.grad.detach().data.clone().flatten())
                    param.grad.zero_()
                current_shared_grad = torch.cat(current_shared_grad, dim=0)

                shared_grad, local_alpha_old, local_alpha_current = GRAD_METHODS[args.mtl](torch.stack([old_shared_grad, current_shared_grad]), args.c)["updating_grad"]
                total_length = 0
                for param in classifier.parameters():
                    length = param.numel()
                    param.grad.data = shared_grad[
                        total_length : total_length + length
                    ].reshape(param.shape)
                    total_length += length

                # params update
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                td.set_postfix(
                    past_loss = np.array(past_losses).mean(),
                    cur_loss = np.array(cur_losses).mean(),
                    cur_acc = total_cur_hits / cur_sampled,
                    ovr_acc = total_hits / sampled,
                )

            return local_alpha_old, local_alpha_current

        alpha_old, alpha_old = 1., 1.
        past_rel_ids = self.relids_of_task[:-1]
        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            combined_data_loader = zip(
                *[get_data_loader(args, [instance for instance in replay_data if instance["relation"] in task_relids], shuffle=True) for task_relids in past_rel_ids],
                get_data_loader(args, current_task_data, shuffle=True)
            )
            alpha_old, alpha_current = train_data(combined_data_loader, e_id, f"{name}{e_id + 1}")

            # SWAG
            all_data = [instance for instance in replay_data if instance["relation"] in flatten_list(past_rel_ids)]
            all_data.extend(current_task_data)
            data_loader = get_data_loader(args, all_data, shuffle=True)
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

        self.past_alphas = [alpha * alpha_old for alpha in self.past_alphas]
        self.past_alphas.append(alpha_current)

    def _train_normal_classifier(self, args, encoder, classifier, past_classifier, swag_classifier, replayed_epochs, current_task_data, name=""):
        encoder.train()
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

        # Validation set
        # validation_data = [instance for instance in flatten_list(replayed_epochs) if instance["relation"] in self.relids_of_task[-1]]
        # validation_data = [instance for instance in flatten_list(replayed_epochs)]
        # validation_data = test_data

        # consecutive_satisfaction = 0
        for e_id in range(args.classifier_epochs):
            replay_data = replayed_epochs[e_id % args.replay_epochs]
            # all_data = [instance for instance in replay_data if instance["relation"] in self.relids_of_task[-1]]
            # all_data.extend(current_task_data)
            data_loader = get_data_loader(args, replay_data, shuffle=True)
            train_data(data_loader, f"{name}{e_id + 1}")
            # swag_classifier.collect_model(classifier)
            # if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
            #     swag_classifier.sample(0.0)
            #     bn_update(data_loader, swag_classifier)

            # Valid (every 5 epochs) and early stop
            # if (e_id + 1) % 5:
            #     continue

            # valid_loss = self._validation(args, classifier, validation_data)
            # if valid_loss >= 0.94:
            #     consecutive_satisfaction += 1
            # else: consecutive_satisfaction = 0
            
            # if consecutive_satisfaction >= 5:
            #     print("EARLY STOP!!!")
            #     break

        return classifier
            
    def train_embeddings(self, args, encoder, final_linear, training_data, task_id):
        encoder.train()
        classifier = Classifier(args=args, final_linear=final_linear).to(args.device)
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
                reps = classifier(encoder_out)

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

    @torch.no_grad()
    def sample_gmm_data(self, args, encoder, encoded_data, name):
        """
        :param encoded_data: (List) data of relation
        """

        encoder.eval()
        data_loader = get_data_loader(args, encoded_data, shuffle=False)
        td = tqdm(data_loader, desc=name)

        # output dict
        out = {}

        # x_data
        x_encoded = []

        for (_, tokens, _) in td:
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            x_encoded.append(tokens)
            # x_encoded.append(encoder(tokens)) # When encoded_data is not encoded but is in original format (tokens)
        x_encoded = torch.cat(x_encoded, dim=0)
        key_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_encoded.cpu().detach().numpy())
        if args.gmm_num_components == 1:
            key_mixture.weights_[0] = 1.0

        out["replay_key"] = key_mixture
        return out

    def sample_vae_data(self, args, encoder, encoded_data, name):
        """
        :param encoded_data: (List) data of relation
        """
        print(name)

        encoder.eval()
        data_loader = get_data_loader(args, encoded_data, shuffle=True)
        
        # Num_training_data
        num_total_train = len(encoded_data)

        # output dict
        out = {}
        vae = GaussianVAE(args).to(args.device)
        vae.fit(
            data_loader=data_loader,
            num_total_train=num_total_train,
            epochs=args.gen_epochs,
            learning_rate=args.gen_lr
        )
        out["replay_key"] = vae
        return out

    def sample_cvae_data(self, args, encoder, encoded_data, name):
        """
        :param encoded_data: (List) data of task
        """
        print(name)

        unique_classes = list(set(
            [item["relation"] for item in encoded_data]
        ))
        glob2task_relid = {}
        for i, class_id in enumerate(unique_classes):
            glob2task_relid[class_id] = i
        
        local_encoded_data = copy.deepcopy(encoded_data)
        for i, item in enumerate(local_encoded_data):
            local_encoded_data[i]["relation"] = glob2task_relid[item["relation"]]

        encoder.eval()
        data_loader = get_data_loader(args, local_encoded_data, shuffle=True)

        # Num_training_data
        num_total_train = len(local_encoded_data)
        
        # output dict
        out = {}
        cvae = ConditionalVAE(args, glob2task_relid).to(args.device)
        cvae.fit(
            data_loader=data_loader,
            num_total_train=num_total_train,
            epochs=args.gen_epochs,
            learning_rate=args.gen_lr
        )

        out["replay_key"] = cvae
        return out

    @torch.no_grad()
    def _validation(self, args, classifier, valid_data):
        classifier.eval()

        valid_data_loader = get_data_loader(args, valid_data, shuffle=False)
        td = tqdm(valid_data_loader, desc="Validating")

        sampled = 0
        total_hits = 0

        # testing
        for (labels, tokens, _) in td:
            sampled += len(labels)
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

            # prediction
            reps = classifier(tokens)
            _, pred = reps.max(1)

            # accuracy_0
            total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

            # display
            td.set_postfix(acc=np.round(total_hits / sampled, 3))
        return total_hits / sampled

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, classifier, test_data, name, task_id):
        # models evaluation mode
        encoder.eval()
        classifier.eval()

        # data loader for test set
        data_loader = get_data_loader(args, test_data, shuffle=False)

        # tqdm
        td = tqdm(data_loader, desc=name)

        # initialization
        sampled = 0
        total_hits = np.zeros(1)

        # testing
        for step, (labels, tokens, _) in enumerate(td):
            sampled += len(labels)
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

            # encoder forward
            encoder_out = encoder(tokens)

            # prediction
            reps = classifier(encoder_out)
            probs = F.softmax(reps, dim=1)
            _, pred = probs.max(1)

            # accuracy_0
            total_hits[0] += (pred == targets).float().sum().data.cpu().numpy().item()

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
            encoder_list = []
            encoder = BertRelationEncoder(config=args).to(args.device)

            # past classifier
            past_classifier = None

            # Top linear
            if args.ETF:
                final_linear = ETFLinear(args.encoder_output_size, args.rel_per_task * args.num_tasks, device=args.device)
            else: final_linear = None

            # initialize memory
            self.memorized_samples = {}

            # load data and start computation
            all_tasks = []
            seen_data = {}

            # Relation ids of every task
            self.relids_of_task = []

            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                print("=" * 100)
                print(f"task={steps+1}")
                print(f"current relations={current_relations}")

                # Live result
                with open(f"live_{args.logname}", "a") as writer:
                    writer.write("=" * 100)
                    writer.write("\n")
                    writer.write(f"task={steps+1}\n")
                    writer.write(f"current relations={current_relations}\n")

                self.steps = steps
                self.not_seen_rel_ids = [rel_id for rel_id in range(args.num_tasks * args.rel_per_task) if rel_id not in [self.rel2id[relation] for relation in seen_relations]]

                # initialize
                cur_training_data = []
                cur_test_data = []
                cur_rel_ids = []

                for i, relation in enumerate(current_relations):
                    cur_training_data += training_data[relation]
                    seen_data[relation] = training_data[relation]
                    cur_test_data += test_data[relation]

                    rel_id = self.rel2id[relation]
                    cur_rel_ids.append(rel_id)
                    self.id2taskid[rel_id] = steps
                self.relids_of_task.append(cur_rel_ids)

                # Classifier
                task_predictor = Classifier(args=args, final_linear=final_linear).to(args.device)

                # train encoder
                if steps == 0:
                    if not args.encoder_checkpoint or not os.path.exists("./ckpt/encoder.pt"):
                        self.train_embeddings(args, encoder, final_linear, cur_training_data, task_id=steps)
                        encoder.encoder.first_task_embeddings = copy.deepcopy(encoder.encoder.embeddings)
                        encoder.encoder.first_task_embeddings.eval()
                        encoder.freeze_embeddings()
                        torch.save(encoder.state_dict(), "./ckpt/encoder.pt")
                    else:
                        print("Load encoder from checkpoint!")
                        encoder.encoder.first_task_embeddings = copy.deepcopy(encoder.encoder.embeddings)
                        encoder.load_state_dict(torch.load("./ckpt/encoder.pt"))

                    for i in range(30):
                        encoder_list.append(copy.deepcopy(encoder).to(args.device))

                # Current encoded data
                cur_training_encoded = convert_data_tokens_to_queries(args, cur_training_data, encoder)

                # memory
                if args.generative != "ConditionalVAE":
                    for i, relation in enumerate(current_relations):
                        relation_encoded_training_data = [x for x in cur_training_encoded if x["relation"] == self.rel2id[relation]]
                        self.memorized_samples[self.rel2id[relation]] =  self.sample_memorized_data(
                                                                                args, encoder,
                                                                                relation_encoded_training_data,
                                                                                f"sampling_relation_{i+1} - {relation}",
                                                                            )
                        rel_id = self.rel2id[relation]
                        replay_key = self.memorized_samples[rel_id]["replay_key"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                        for e_id in range(args.replay_epochs):
                            for x_encoded in replay_key[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                                self.replayed_key[e_id].append({"relation": rel_id, "tokens": x_encoded})
                else:
                    self.memorized_samples[steps] = self.sample_memorized_data(
                                                        args,
                                                        encoder,
                                                        cur_training_encoded,
                                                        f"Fitting_task_{steps+1}",
                                                    )
                    for i, relation in enumerate(current_relations):
                        rel_id = self.rel2id[relation]
                        replay_key = self.memorized_samples[steps]["replay_key"].sample(args.replay_epochs * args.replay_s_e_e, label=rel_id)[0].astype("float32")
                        for e_id in range(args.replay_epochs):
                            for x_encoded in replay_key[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                                self.replayed_key[e_id].append({"relation": rel_id, "tokens": x_encoded})
                        print(f"Done sampling relation {i+1}: {relation}")

                # all test data
                all_tasks.append(cur_test_data)

                # swag task predictor
                swag_task_predictor = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

                # train
                if steps == 0:
                    past_classifier = self._train_normal_classifier(args, encoder, task_predictor, None, swag_task_predictor, self.replayed_key, None, "train_task_predictor_epoch_")
                else:
                    past_classifier = self.train_classifier(args, encoder, task_predictor, past_classifier, swag_task_predictor, self.replayed_key, cur_training_encoded, "train_task_predictor_epoch_")

                # prediction
                print("===NON-SWAG===")
                with open(f"live_{args.logname}", "a") as writer:
                    writer.write("===NON-SWAG===\n")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([len(i_th_test_data), self.evaluate_strict_model(args, encoder, task_predictor, i_th_test_data, f"test_task_{i+1}", steps)])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                cur_task_acc = [result[1] for result in results]
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                with open(f"live_{args.logname}", "a") as writer:
                    writer.write(f"current test accuracy: {cur_acc}\n")
                    writer.write(f"history test accuracy: {total_acc}\n")
                    writer.write(f"all history accuracy: {cur_task_acc}\n")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                print("===UNTIL-NOW==")
                writer = open(f"live_{args.logname}", "a")
                writer.write("===UNTIL-NOW==\n")
                print("accuracies:")
                writer.write("accuracies:\n")
                for x in test_cur:
                    print(x)
                    writer.write(f"{x}\n")
                print("averages:")
                writer.write("averages:\n")
                for x in test_total:
                    print(x)
                    writer.write(f"{x}\n")
                writer.close()
