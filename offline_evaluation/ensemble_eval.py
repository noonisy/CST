import torch
import sys
sys.path.append('../')
import json
import pickle
import evaluation
import argparse
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from data import COCO, DataLoader
from data import ImageDetectionsField, TextField, RawField
from evaluation import Cider
from models.CST import Transformer, TransformerEncoder, TransformerDecoderLayer, TransformerEnsemble


def predict_captions(model, dataloader, text_field, cider, args):
    import itertools
    tokenizer_pool = mp.Pool(4)
    examp = {}
    res = {}
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            if args.examples:
                caps_gen1 = text_field.decode(out)
                caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

                caps_gen1, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
                caps_gt1 = evaluation.PTBTokenizer.tokenize(caps_gt1)
                caps_gen1 = evaluation.PTBTokenizer.tokenize(caps_gen1)
                reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)

                for i, (gts_i, gen_i) in enumerate(zip(caps_gt1, caps_gen1)):
                    examp[len(examp)] = {
                        'gt': caps_gt1[gts_i],
                        'gen': caps_gen1[gen_i],
                        'cider': reward[i].item(),
                    }

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, all_scores = evaluation.compute_scores(gts, gen)
    tokenizer_pool.close()
    if args.examples:
        json.dump(examp, open(args.examples_json, 'w'))
    if args.scores:
        res['bleu1'] = all_scores['BLEU'][0]
        res['bleu4'] = all_scores['BLEU'][3]
        res['METEOR'] = all_scores['METEOR']
        res['ROUGE'] = all_scores['ROUGE'].tolist()
        res['CIDEr'] = all_scores['CIDEr'].tolist()
        # res['SPICE'] = all_scores['SPICE']
        json.dump(res, open(args.scores_json, 'w'))
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='CST')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--features_type', type=str, default='X101', choices=['X101', 'X152'])
    parser.add_argument('--features_path', type=str, default='/home/noonisy/local/X101_grid_feats_coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/home/noonisy/data/annotations')
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
    parser.add_argument('--examples', action='store_true', default=False)
    parser.add_argument('--examples_json', type=str, default='CST_examples.json')
    parser.add_argument('--scores', action='store_true', default=False)
    parser.add_argument('--scores_json', type=str, default='CST_scores.json')

    args = parser.parse_args()
    print(args)
    print('CST Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=args.max_detections, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('../vocab.pkl', 'rb'))

    # Prepare cider evaluation
    ref_caps_test = test_dataset.text()
    cider_test = Cider(evaluation.PTBTokenizer.tokenize(ref_caps_test))

    # Prepare dataloader
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    # Model
    encoder = TransformerEncoder(args.N_enc, padding_idx=0, d_in=args.d_ff, d_model=args.d_model, d_k=args.d_k, d_v=args.d_v,
                                 h=args.head, d_ff=args.d_ff, dropout=0.1, M=args.M, p=args.p)
    decoder = TransformerDecoderLayer(vocab_size=len(text_field.vocab), max_len=54, N_dec=args.N_dec, padding_idx=text_field.vocab.stoi['<pad>'],
                                      d_model=args.d_model, d_k=args.d_k, d_v=args.d_v, h=args.head, d_ff=args.d_ff, dropout=0.1)
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    model_path = [
        f'../ensemble_models/CST_{args.features_type}_1.pth',
        f'../ensemble_models/CST_{args.features_type}_2.pth',
        f'../ensemble_models/CST_{args.features_type}_3.pth',
        f'../ensemble_models/CST_{args.features_type}_4.pth'
    ]

    ensemble_model = TransformerEnsemble(model=model, weight_files=model_path)

    scores = predict_captions(ensemble_model, dict_dataloader_test, text_field, cider_test, args)
    print(scores)
    print('finished!')
