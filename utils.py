# utility functions such as statistics

import json
import argparse


def get_subset_stats(json_path):
    with open(json_path) as json_file:
        data_index = json.load(json_file)
    stats = {}
    for subset in ['train','test']:
        stats[subset] = {'Cancer': len([k for k,v in data_index.items() if (v['subset'] == subset) and (v['cancer'] == True)]),
                        'No Cancer': len([k for k,v in data_index.items() if (v['subset'] == subset) and (v['cancer'] == False)])}

    print("{:<8} {:<8} {:<10} {:<8}".format('Subset', 'Total', 'Cancerous', 'Non-cancerous'))
    for k, v in stats.items():
        cancer = v['Cancer']
        non_cancer = v['No Cancer']
        print("{:<8} {:<8} {:<10} {:<8}".format(k, cancer+non_cancer,cancer, non_cancer))


def metrics_summary(metrics_path):
    print(
        "{:<12} {:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format('Model', 'Dataset', 'avg_dice', 'c_dice',
                                                                               'n_dice', 'precision', 'recall',
                                                                               'overlap', 'FPR', 'FNR'))
    for file in os.listdir(metrics_path):
        file_path = os.path.join(metrics_path, file)
        with open(file_path) as json_file:
            metrics = json.load(json_file)

        dataset = metrics['train']['dataset']
        model = metrics['train']['model']
        image_size = metrics['train']['image_size']
        avg_dice = metrics['test']['average_dice_score']
        cancer_dice = metrics['test']['average_cancer_dice_score']
        no_cancer_dice = metrics['test']['average_non_cancer_dice_score']

        FP = metrics['test']['gt_n_pd_c']
        TP = metrics['test']['gt_c_pd_c_overlap'] + metrics['test']['gt_c_pd_c_no_overlap']
        FN = metrics['test']['gt_c_pd_no_c']
        TN = metrics['test']['gt_n_pd_n']

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)

        if TP == 0:
            TP_with_overlap = 0
        else:
            TP_with_overlap = metrics['test']['gt_c_pd_c_overlap'] / TP

        false_positive = FP / (FP + TN)
        false_negative = FN / (FN + TP)

        print("{:<12} {:<15} {:.3f}    {:.3f}  {:.3f}  {:.3f}     {:.3f}  {:.3f}   {:.3f} {:.3f}".format(model, dataset,
                                                                                                         avg_dice,
                                                                                                         cancer_dice,
                                                                                                         no_cancer_dice,
                                                                                                         precision,
                                                                                                         recall,
                                                                                                         TP_with_overlap,
                                                                                                         false_positive,
                                                                                                         false_negative))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility funcitons for statistics on the dataset or analysis of metrics"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="util function to be executed",
    )
    parser.add_argument(
        "--jsonfile", type=str, default="./data/data_index_subsets.json",
        help="root folder with json with assigned subsets"
    )
    parser.add_argument(
        "--metrics_path", type=str, default="./save/metrics",
        help="root folder with json with assigned subsets"
    )
    args = parser.parse_args()
    if args.method == 'subset_stats':
        get_subset_stats(args.jsonfile)
    elif args.method == 'metrics_summary':
        metrics_summary(args.metrics_path)
