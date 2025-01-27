import json
import os
import argparse

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def webqsp_evaluate_valid_results(pred_file, gold_file):
    res = main(pred_file, gold_file)
    dirname = os.path.dirname(pred_file)
    filename = os.path.basename(pred_file)
    with open (os.path.join(dirname,f'{filename}_final_eval_results_official.txt'),'w') as f:
        f.write(res)
        f.flush()
        
def graphq_evaluate_valid_results(pred_file, gold_file):
    res = main_graphq(pred_file, gold_file)
    dirname = os.path.dirname(pred_file)
    filename = os.path.basename(pred_file)
    with open (os.path.join(dirname,f'{filename}_final_eval_results_official.txt'),'w') as f:
        f.write(res)
        f.flush()

def FindInList(entry,elist):
    for item in elist:
        if entry == item:
            return True
    return False
            
def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0, 1]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0, 0.0, 1]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList)==0:
        return [1.0, 0.0, 0.0, 0]    # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist =goldAnswerList
        plist =predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry,plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry,glist):
                fp += 1


        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        
        f1 = (2*precision*recall)/(precision+recall)
        
        if tp > 1e-40:
            hit = 1
        else:
            hit = 0
        return [precision, recall, f1, hit]


def main(pred_data, dataset_data):

    goldData = load_json(dataset_data)
    predAnswers = load_json(pred_data)
    assert len(goldData) == len(predAnswers)
    
    PredAnswersById = {}

    for item in predAnswers:
        PredAnswersById[item["ID"]] = item["pred_answer"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hitSum = 0
    numCorrect = 0
    prediction_res = []
    
    goldRaw = load_json('dataset/WebQSP/origin/WebQSP.test.json')
    gold_dict = {d["QuestionId"]: d for d in goldRaw['Questions']}

    for entry in goldData:
        
        total += 1
    
        id = entry["ID"]
    
        if id not in PredAnswersById:
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        predAnswers = PredAnswersById[id]
        
        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        besthit = 0
        entry_raw = gold_dict[entry["ID"]]
        for pidx in range(0,len(entry_raw["Parses"])):
            pidxAnswers = [item['AnswerArgument'] for item in entry_raw["Parses"][pidx]["Answers"]]
            prec,rec,f1,hit = CalculatePRF1(pidxAnswers,predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec
            if hit > besthit:
                besthit = hit

        # pidxAnswers = entry["answer"]
        # prec,rec,f1,hit = CalculatePRF1(pidxAnswers,predAnswers)
        # bestf1 = f1
        # bestf1Rec = rec
        # bestf1Prec = prec
        # besthit = hit

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hitSum += besthit

        pred = entry
        # pred['qid'] = id
        pred['precision'] = bestf1Prec
        pred['recall'] = bestf1Rec
        pred['f1'] = bestf1
        pred['hit'] = besthit
        
        if bestf1 == 1.0:
            numCorrect += 1
            pred['answer_acc'] = True
        else:
            pred['answer_acc'] = False
            
        prediction_res.append(pred)

    print("Average precision over questions: %.6f" % (precSum / total))
    print("Average recall over questions: %.6f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.6f" % (f1sum / total))
    print("F1 of average recall and average precision: %.6f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print("True accuracy (ratio of questions answered exactly correctly): %.6f" % (numCorrect / total))
    print("Hits@1 over questions: %.6f" % (hitSum / total))
    res = f'Average precision over questions: {(precSum / total)}\n, Average recall over questions: {(recSum / total)}\n, Average f1 over questions (accuracy): {(f1sum / total)}\n, F1 of average recall and average precision: {(2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total))}\n, True accuracy (ratio of questions answered exactly correctly): {(numCorrect / total)}\n, Hits@1 over questions: {(hitSum / total)}'
    dirname = os.path.dirname(pred_data)
    filename = os.path.basename(pred_data)
    dump_json(prediction_res, os.path.join(dirname, f'{filename}_new.json'))
    return res


def main_graphq(pred_data, dataset_data):

    goldData = load_json(dataset_data)
    predAnswers = load_json(pred_data)
    assert len(goldData) == len(predAnswers)
    
    PredAnswersById = {}

    for item in predAnswers:
        PredAnswersById[item["ID"]] = item["pred_answer"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hitSum = 0
    numCorrect = 0
    prediction_res = []

    for entry in goldData:

        total += 1
    
        id = entry["ID"]
    
        if id not in PredAnswersById:
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        predAnswers = PredAnswersById[id]

        pidxAnswers = entry["answer"]
        prec,rec,f1,hit = CalculatePRF1(pidxAnswers,predAnswers)
        bestf1 = f1
        bestf1Rec = rec
        bestf1Prec = prec
        besthit = hit

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hitSum += besthit

        pred = entry
        # pred['qid'] = id
        pred['precision'] = bestf1Prec
        pred['recall'] = bestf1Rec
        pred['f1'] = bestf1
        pred['hit'] = besthit
        
        if bestf1 == 1.0:
            numCorrect += 1
            pred['answer_acc'] = True
        else:
            pred['answer_acc'] = False
            
        prediction_res.append(pred)

    print("Average precision over questions: %.6f" % (precSum / total))
    print("Average recall over questions: %.6f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.6f" % (f1sum / total))
    print("F1 of average recall and average precision: %.6f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print("True accuracy (ratio of questions answered exactly correctly): %.6f" % (numCorrect / total))
    print("Hits@1 over questions: %.6f" % (hitSum / total))
    res = f'Average precision over questions: {(precSum / total)}\n, Average recall over questions: {(recSum / total)}\n, Average f1 over questions (accuracy): {(f1sum / total)}\n, F1 of average recall and average precision: {(2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total))}\n, True accuracy (ratio of questions answered exactly correctly): {(numCorrect / total)}\n, Hits@1 over questions: {(hitSum / total)}'
    dirname = os.path.dirname(pred_data)
    filename = os.path.basename(pred_data)
    dump_json(prediction_res, os.path.join(dirname, f'{filename}_new.json'))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="split to operate on, can be `test`, `dev` and `train`",
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="prediction results file"
    )
    args = parser.parse_args()
    
    webqsp_evaluate_valid_results(args)