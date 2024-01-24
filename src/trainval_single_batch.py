import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import json
tao = 0.04

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores



def downstream_task_forward(model, batch,  criterion, args, dataset=None, temp_result_dict=None):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """
    if args.dataset == 'peir':
        y_gt_bce = batch['labels_matrix']
        preds = model(batch)
        loss = criterion(preds, y_gt_bce)
        scores = compute_score_with_logits(preds, y_gt_bce)
        accs = torch.sum(scores, dim=1)

    else:
        x_txt_cands_mc = batch['text_cands_features']
        y_gt_mc = batch['labels_id']

        output = model(batch)

        try:
            merge_features, topk_caption = output
        except:
            merge_features = output

        y_pred = F.cosine_similarity(merge_features.unsqueeze(1), x_txt_cands_mc, dim=-1)  # (N, N_ans)
        loss = criterion(y_pred / tao, y_gt_mc)

        if model.training:
            accs = (y_pred.argmax(dim=-1) == y_gt_mc).float()
            return loss, accs

    l_y_pred = list(y_pred.argmax(dim=-1).cpu().numpy())
    l_y_gt = list(y_gt_mc.cpu().numpy())
    vids, quess, qids, anstype = batch['additional_info']

    count = 0
    count_true = 0
    for i, (vid, ques, qid, i_pred, i_gt) in enumerate(zip(vids, quess, qids, l_y_pred, l_y_gt)):
        count += 1
        pred = dataset.label2answer[i_pred]
        ans = dataset.label2answer[i_gt]

        if pred in ans:
            result = True
            count_true += 1
        else:
            result = False

        # if args.visible:
            # topk_indict = [int(item) if isinstance(item, np.integer) else item for item in topk_caption[i]]
            # temp_result_dict.append(
            #         {
            #             "img_id": vid,
            #             "qid": int(qid),
            #             "question": ques,
            #             "answer_type": anstype[i],
            #             "answer": ans,
            #             "prediction": pred,
            #             "result": result,
            #             "tok_indices": topk_indict
            #         }
            #     )
            # line = f"{vid}\t{ques}\t{anstype[i]}\t{pred}\t{ans}\t{result}"
            # print(line)


    return loss, count, count_true

