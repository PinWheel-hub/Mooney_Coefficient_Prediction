import torch
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_one_epoch(model_train, model, loss_fn, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period):
    loss = 0
    val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            tensors1, tensors2, targets = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    tensors1 = tensors1.cuda()
                    tensors2 = tensors2.cuda()
                    targets = targets.cuda()
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(tensors1, tensors2)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = loss_fn(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'average loss': loss / (iteration + 1),
                                'current loss': loss_value.item(),
                                'output': outputs[0].item(),
                                'target': targets[0].item(),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            tensors1, tensors2, targets = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    tensors1 = tensors1.cuda()
                    tensors2 = tensors2.cuda()
                    targets = targets.cuda()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(tensors1, tensors2)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = loss_fn(outputs, targets)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
