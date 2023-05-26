import os
import pickle
import random
import itertools
import evaluation
import multiprocessing
import torch
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from torch.nn import NLLLoss
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from evaluation import PTBTokenizer, Cider
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from models.CST import Transformer, TransformerEncoder, TransformerDecoderLayer


def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def evaluate_loss(model, dataloader, loss_fn, text_field, e):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % (e+1), unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss='%.4f' % (running_loss / (it + 1)))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, e):
    # Validation metrics
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % (e+1), unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, scheduler, text_field, loss_fn, e):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    print('lr = ', optim.state_dict()['param_groups'][0]['lr'])

    with tqdm(desc='Epoch %d - train' % (e+1), unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss='%.4f' % (running_loss / (it + 1)))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, scheduler, cider, text_field, e):
    # Training with SCST
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    scheduler.step()
    running_loss = .0
    seq_len = 20
    beam_size = 5
    print('lr = ', optim.state_dict()['param_groups'][0]['lr'])

    with tqdm(desc='Epoch %d - train' % (e+1), unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True, dtype=torch.float32)

            avg_log_probs = torch.sum(log_probs, -1) / torch.sum(log_probs != 0, -1)
            loss = -avg_log_probs * (reward - reward_baseline)
            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward='%.4f' % (running_reward / (it + 1)),
                             reward_baseline='%.4f' % (running_reward_baseline / (it + 1)))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    tokenizer_pool.close()
    return loss, reward, reward_baseline


def train(args):
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=args.max_detections, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    # load vocabulary
    print('Loading from vocabulary')
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    print(len(text_field.vocab))

    # Model
    encoder = TransformerEncoder(args.N_enc, padding_idx=0, d_in=args.d_ff, d_model=args.d_model, d_k=args.d_k, d_v=args.d_v,
                                 h=args.head, d_ff=args.d_ff, dropout=0.1, M=args.M, p=args.p)
    decoder = TransformerDecoderLayer(vocab_size=len(text_field.vocab), max_len=54, N_dec=args.N_dec, padding_idx=text_field.vocab.stoi['<pad>'],
                                      d_model=args.d_model, d_k=args.d_k, d_v=args.d_v, h=args.head, d_ff=args.d_ff, dropout=0.1)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    # Model parameters
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print('Number of trainable parameters: %.2fM' % (total / 1e6))

    # dataloaders
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = train_dataset.text()
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    # learning rate (xe)
    def lambda_lr(s):
        base_lr = 0.0001
        print("s:", s)
        if s <= 3:
            lr = base_lr * s / 4
        elif s <= 10:
            lr = base_lr
        elif s <= 12:
            lr = base_lr * 0.2
        elif s <= 15:
            lr = base_lr * 0.2 * 0.2
        else:
            lr = base_lr * 0.2 * 0.2 * 0.2
        return lr

    # learning rate (scst)
    def lambda_lr_rl(s):
        print("s:", s)
        # if s <= args.rl_at + 8:
        #     lr = 3e-5
        # elif s <= args.rl_at + 18:
        #     lr = 3e-5 * 0.2
        # elif s <= args.rl_at + 28:
        #     lr = 3e-5 * 0.2 * 0.2
        # elif s <= args.rl_at + 38:
        #     lr = 3e-5 * 0.2 * 0.2 * 0.2
        # else:
        #     lr = 3e-5 * 0.2 * 0.2 * 0.2 * 0.2
        if s <= 28:
            lr = 3e-5
        elif s <= 33:
            lr = 6e-6
        elif s <= 40:
            lr = 1e-6
        elif s <= 60:
            lr = 5e-7
        else:
            lr = 5e-8
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    # load states and weights
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            use_rl = data['use_rl']

            if not use_rl:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            else:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, and best_test_cider %f' % (
                data['epoch'] + 1, data['val_loss'], data['best_cider'], data['best_test_cider']))
            print('patience:', data['patience'], 'use_rl', data['use_rl'])
        else:
            print('load model failed')

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True, pin_memory=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.rl_batch_size // 5, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.rl_batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.rl_batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, scheduler, text_field, loss_fn, e)
            writer.add_scalar('data/train_loss', train_loss, e)
            learningrate = optim.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('data/learning_rate', learningrate, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, scheduler_rl, cider_train, text_field, e)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
            learningrate = optim_rl.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('data/learning_rate', learningrate, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, e)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, e)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if not use_rl and (patience >= 20 or e == args.rl_at):
            use_rl = True
            switch_to_rl = True
            patience = 0

            # optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
            # scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

            for _ in range(e + 1):
                scheduler_rl.step()
            print("Switching to RL")

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }, os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name))

        if best:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
        if best_test:
            copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_best_test.pth' % args.exp_name))

        # if switch_to_rl:
        #     copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_xe_last.pth' % args.exp_name))
        #     copyfile(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '%s_xe_best.pth' % args.exp_name))

        # if e >= 40:
        #     copyfile(os.path.join(args.dir_to_save_model, '%s_last.pth' % args.exp_name), os.path.join(args.dir_to_save_model, '{}_epoch_{}.pth'.format(args.exp_name, e+1)))

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.dir_to_save_model, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f, and best test_cider %f' % (
                data['epoch'] + 1, data['val_loss'], data['best_cider'], data['best_test_cider']))

        if e >= 69:
            print('patience reached.')
            exit_train = True

        if exit_train:
            writer.close()
            break


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='CST')
    parser.add_argument('--exp_name', type=str, default='CST')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--rl_batch_size', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_in', type=int, default=2048)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--max_detections', type=int, default=49)
    parser.add_argument('--rl_at', type=int, default=3)
    parser.add_argument('--seed', type=int, default=555555)
    parser.add_argument('--warmup', type=int, default=10000)

    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str, default='/home/noonisy/local/X101_grid_feats_coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/home/noonisy/data/annotations')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--dir_to_save_model', type=str, default='checkpoints/')

    args = parser.parse_args()
    print(args)

    setseed(args.seed)
    print(f'{args.exp_name} Training')

    train(args)
