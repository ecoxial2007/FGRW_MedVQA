"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""
import json
import os
import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate

class VQADataset(data.Dataset):
    """
    Example (simple) video-language dataset class, for illustration purposes for ATP training.
    """
    def __init__(self, args, split="train"):
        super().__init__()
        self.anno_path = args.anno_path
        self.dataset = os.path.split(self.anno_path)[-1]


        self.data_path = args.data_path
        self.split = split
        self.visible = args.visible

        with open(os.path.join(self.anno_path, f'{split}_en.json'), 'r') as jf:
            self.metadata = json.load(jf)


        self.clip = ''
        self.text_features = h5py.File(os.path.join(self.data_path, f'text_features{self.clip}.h5'), 'r')
        self.text_cands_features = torch.tensor(self.text_features['label_features'], dtype=torch.float32)


        with open(os.path.join(self.anno_path, f'ans2label_en.json'), 'r') as jf:
            self.answer2label = json.load(jf)

        self.label2answer = {value: key for key, value in self.answer2label.items()}

        image_ids = []
        for f in self.metadata:
            image_ids.append(f['img_id'])
        self.image_ids = list(set(image_ids))

        print(self.split, len(self.metadata))
        print('num_classes', len(self.answer2label))
        print('num_images', len(self.image_ids))


    def __len__(self):
        return len(self.metadata)




    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """
        f = self.metadata[index]
        qid = int(f['qid'])
        image_id = f['img_id']
        image_id = os.path.splitext(image_id)[0]
        question = f['question']
        answer = str(f['answer'])
        answer_type = str(f['answer_type'])

        labels_id = torch.tensor(self.answer2label[answer], dtype=torch.long)

        if self.dataset == 'PathVQA':
            image_path = os.path.join(self.data_path, 'images', image_id.split('_')[0], image_id+'.jpg')
        else:
            image_path = os.path.join(self.data_path, 'images', image_id+'.jpg')

        image_feature_path = image_path.replace('images', f'features{self.clip}').replace('.jpg', '.h5')
        image_features_file = h5py.File(image_feature_path, 'r')

        image_features = torch.tensor(np.array(image_features_file['feature']), dtype=torch.float32)  # (L_video, D_in); L_video >> L
        patch_features = torch.tensor(np.array(image_features_file['feature_noproj']), dtype=torch.float32)  # (L_video, D_in); L_video >> L

        text_query_features = torch.tensor(self.text_features['question_tokens'][qid], dtype=torch.float32)
        text_query_features_global = torch.tensor(self.text_features['question_features'][qid], dtype=torch.float32)

        item_dict = {
            'image_features': image_features,
            'patch_features': patch_features,
            'text_query_features': text_query_features_global,
            'text_query_token_features': text_query_features,
            'text_cands_features': self.text_cands_features,
            'labels_id': labels_id,
        }

        if self.visible:
            item_dict.update({
                'additional_info': (image_id, question, qid, answer_type),
            })

        return item_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Parser for FGRA training script.")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size")
    parser.add_argument('--num_workers', default=1, type=int, help="number of dataset workers")
    parser.add_argument('--anno_path', type=str, help='Annotation', default='./data/Annotations/VQA-RAD')
    parser.add_argument('--data_path', type=str, help='Feature', default='./data/VQA-RAD')
    parser.add_argument('--visible', type=bool, default=False) #for check each question predicts answer
    args = parser.parse_args()
    dataset = VQADataset(args=args)

    dldr_train = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    for i, batch in enumerate(dldr_train):
        print(batch)
        quit()