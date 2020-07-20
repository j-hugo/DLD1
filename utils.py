# utility functions such as statistics

import json


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the model for image segmentation of Colon"
    )
    parser.add_argument(
        "--function",
        type=str,
        default=None,
        help="util function to be executed",
    )
    parser.add_argument(
        "--jsonfile", type=str, default="./data/data_index_subsets.json",
        help="root folder with json with assigned subsets"
    )
    args = parser.parse_args()
    if args.method == 'subset_stats':
        get_subset_stats(args.jsonfile)
