import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, Adaptive_Gaussian_Dice_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score

import math


def Dynamic_Hybrid_Loss(inputs, target, cls_weights=None, num_classes=8,
                        epoch=0, total_epochs=200, base_sigma=2.0,
                        min_alpha=0.4, max_alpha=0.8):
    """
    Dynamic Weighted Loss:
    - Early training (high alpha) => CE dominant (stable convergence)
    - Late training (low alpha) => Dice dominant (enhanced boundary and drift resistance)
    """
    # Linear decay of alpha
    alpha = max_alpha - (max_alpha - min_alpha) * (epoch / total_epochs)

    ce_loss = CE_Loss(inputs, target, cls_weights, num_classes)
    dice_loss = Adaptive_Gaussian_Dice_Loss(
        inputs, F.one_hot(target, num_classes=num_classes).float(),
        base_sigma=base_sigma
    )

    total_loss = alpha * ce_loss + (1 - alpha) * dice_loss
    return total_loss, ce_loss.item(), dice_loss.item(), alpha


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, agd_dice_loss, cls_weights, num_classes, save_period,
                  save_dir, log_dir, local_rank=0, eval_period=5, base_sigma=2.0,
                  min_alpha=0.4, max_alpha=0.8):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # ----------------------#
        #   Forward propagation
        # ----------------------#
        outputs = model_train(imgs)

        # ----------------------#
        #   Calculate loss
        # ----------------------#
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if agd_dice_loss:
            epoch_in = 0  # Note: Hardcoded 0 is used here as provided. Modify if dynamic change is needed

            progress = epoch_in / max(1, epoch)
            alpha = max_alpha - (max_alpha - min_alpha) * progress
            alpha = max(min_alpha, min(max_alpha, alpha))  # Limit range

            loss_agd = Adaptive_Gaussian_Dice_Loss(outputs, labels, base_sigma=base_sigma)
            loss = alpha * loss + (1 - alpha) * loss_agd

        with torch.no_grad():
            # -------------------------------#
            #   Calculate f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   Forward propagation
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   Calculate loss
            # ----------------------#
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if agd_dice_loss:
                epoch_in = 0

                progress = epoch_in / max(1, epoch)
                alpha = max_alpha - (max_alpha - min_alpha) * progress
                alpha = max(min_alpha, min(max_alpha, alpha))  # Limit range

                loss_agd = Adaptive_Gaussian_Dice_Loss(outputs, labels, base_sigma=base_sigma)
                loss = alpha * loss + (1 - alpha) * loss_agd
            # -------------------------------#
            #   Calculate f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   Save weights
        # -----------------------------------------------#

        epoch_1 = epoch + 1

        if (epoch_1 % eval_period) == 0:

            with open(os.path.join(log_dir, "epoch_miou.txt"), 'r') as file:
                lines = file.readlines()

                last_line = lines[-1].strip() if lines else None
                l = len(lines)
                last_l = l - 1
                if len(lines) > 2:
                    sub_list = [item.strip() for item in lines] if lines else None
                    max_value = max(sub_list[:last_l])
                    if float(last_line) > float(max_value):
                        print('Save best model to best_epoch_weights.pth')

                        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda,
                         agd_dice_loss, cls_weights, num_classes, save_period, save_dir, local_rank=0,
                         base_sigma=2.0, min_alpha=0.4, max_alpha=0.8):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # ----------------------#
        #   Forward propagation
        # ----------------------#
        outputs = model_train(imgs)
        # ----------------------#
        #   Calculate loss
        # ----------------------#
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if agd_dice_loss:
            epoch_in = 0

            progress = epoch_in / max(1, epoch)
            alpha = max_alpha - (max_alpha - min_alpha) * progress
            alpha = max(min_alpha, min(max_alpha, alpha))  # Limit range

            loss_agd = Adaptive_Gaussian_Dice_Loss(outputs, labels, base_sigma=base_sigma)
            loss = alpha * loss + (1 - alpha) * loss_agd

        with torch.no_grad():
            # -------------------------------#
            #   Calculate f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        # -----------------------------------------------#
        #   Save weights
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))