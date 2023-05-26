import json
import torch
import pickle
import argparse
import itertools
import sys
sys.path.append('../')
from data import TextField
from dataset import Online_Test
from torch.utils.data import DataLoader
from models.CST import Transformer, TransformerEncoder, TransformerDecoderLayer, TransformerEnsemble


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='CST')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
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
    # parser.add_argument('--max_detections', type=int, default=49)

    parser.add_argument('--benchmark', type=str, default='COCO', choices=['COCO', 'nocaps'])
    parser.add_argument('--task_type', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--features_type', type=str, default='X152', choices=['X101', 'X152'])
    parser.add_argument('--features_path', type=str, default='/home/noonisy/local/X101_grid_feats_coco_test.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/home/noonisy/data/annotations/image_info_test2014.json')

    args = parser.parse_args()
    print(args)
    print('CST Evaluation')
    output_path = f'captions_{args.task_type}2014_CST{args.features_type}_results.json'

    # if args.features_type == 'X152':
    args.models_path = [
        f'../ensemble_models/CST_{args.features_type}_1.pth',
        f'../ensemble_models/CST_{args.features_type}_2.pth',
        f'../ensemble_models/CST_{args.features_type}_3.pth',
        f'../ensemble_models/CST_{args.features_type}_4.pth'
    ]

    if args.benchmark == 'nocaps':
        output_path = f'captions_nocaps_val_CST{args.features_type}_results.json'
        args.model_path = f'../checkpoints/CST_{args.features_type}.pth'

    print(output_path)

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('../vocab.pkl', 'rb'))

    dataset = Online_Test(feat_path=args.features_path, ann_file=args.annotation_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # Model
    encoder = TransformerEncoder(args.N_enc, padding_idx=0, d_in=args.d_ff, d_model=args.d_model, d_k=args.d_k, d_v=args.d_v,
                                 h=args.head, d_ff=args.d_ff, dropout=0.1, M=args.M, p=args.p)
    decoder = TransformerDecoderLayer(vocab_size=len(text_field.vocab), max_len=54, N_dec=args.N_dec, padding_idx=text_field.vocab.stoi['<pad>'],
                                      d_model=args.d_model, d_k=args.d_k, d_v=args.d_v, h=args.head, d_ff=args.d_ff, dropout=0.1)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    if args.benchmark == 'COCO':
        model = TransformerEnsemble(model=model, weight_files=args.models_path)
    else:
        model.load_state_dict(torch.load(args.model_path)['state_dict'])

    # generate captions
    model.eval()
    outputs = []
    for it, (image_ids, images) in enumerate(iter(dataloader)):
        if it % 100 == 0:
            print('processing {} / {}'.format(it, len(dataset) // args.batch_size))
        images = images.to(device)
        with torch.no_grad():
            out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        caps_gen = [' '.join([k for k, g in itertools.groupby(gen_i)]).strip() for gen_i in caps_gen]
        for i in range(image_ids.size(0)):
            item = {}
            item['image_id'] = int(image_ids[i])
            item['caption'] = caps_gen[i]
            outputs.append(item)

    with open(output_path, 'w') as f:
        json.dump(outputs, f)

    print(f'finished, saved as {output_path}')
