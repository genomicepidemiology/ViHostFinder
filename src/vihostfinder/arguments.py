import argparse


class VArguments:


    @staticmethod
    def select_partitions(partition):
        if partition == 1:
            train_part = [1,2,3]
            val_part = [4]
            test_part = [5]
        elif partition == 2:
            train_part = [2,3,4]
            val_part = [5]
            test_part = [1]
        elif partition == 3:
            train_part = [3,4,5]
            val_part = [1]
            test_part = [2]
        elif partition == 4:
            train_part = [4,5,1]
            val_part = [2]
            test_part = [3]
        else:
            train_part = [5,1,2]
            val_part = [3]
            test_part = [4]        
        return train_part, val_part, test_part

    @staticmethod
    def select_filedir(lm, sampler):
        if lm == "hyenadna":
            if sampler:
                file_dir = "/work3/alff/ViralInf/RNA_db/data/embeddingvector/embeddingvector_all_hyenadna/"
            else:
                file_dir = "/work3/alff/ViralInf/RNA_db/data/embeddingvector/embeddingvector_all_hyenadna/"
        else:
            file_dir = "/work3/alff/ViralInf/RNA_db/data/embeddingvector/embeddingvector_caduceus/"
        return file_dir

    @staticmethod
    def get_annotationfile(sampler):
        if not sampler:
            filepath = "/work3/alff/ViralInf/RNA_db/data/metadata/repr_dbredux_part2.tsv"
        elif sampler == "Cluster0.4":
            filepath = "/work3/alff/ViralInf/RNA_db/data/metadata/repr_db_origlables_partition04.tsv"
        elif sampler == "Cluster0.45":
            filepath = "/work3/alff/ViralInf/RNA_db/data/metadata/repr_db_origlables_partition045.tsv"
        else:
            raise ValueError("Sampler {} not available".format(sampler))
        return filepath

    @staticmethod
    def fix_layers(layer_str):
        return list(map(int, layer_str.split(',')))

    @staticmethod
    def create_arguments():

        # Parent parser for shared arguments
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument("--out")
        parent_parser.add_argument('--lm', help='lm', choices=["hyenadna", "caduceus"])
        parent_parser.add_argument('--labels', help='labels', choices=["orig_flat", "orig_hier1", "orig_hier2"])
        parent_parser.add_argument("--model", help="model", choices=["flatNN", "HMCNF"])
        parent_parser.add_argument("--beta", default=0.5, type=float)
        parent_parser.add_argument("--global_layers", default="128,128,256,64")
        parent_parser.add_argument("--local_layers", default="64,64,128,32")
        parent_parser.add_argument("--annotation_file", default=False)

        # Main parser
        parser = argparse.ArgumentParser(
            description='Description of your program',
            parents=[parent_parser]
        )

        subparsers = parser.add_subparsers(dest="command", required=True)

        # Train subparser
        train_parser = subparsers.add_parser("train", help="Train ViHostFinder", parents=[parent_parser])
        train_parser.add_argument('--partition', help='labels', choices=[1,2,3,4,5], type=int)
        train_parser.add_argument("--lr", default=0.001, type=float)
        train_parser.add_argument("--dropout", default=0.2)
        train_parser.add_argument("--sigma", default=0.2, type=float)
        train_parser.add_argument("--cluster_frequency", default=0., type=float)
        train_parser.add_argument("--sampler_cluster", default=False, choices=[False, "Cluster0.4", "Cluster0.45"])
        train_parser.add_argument("--wandb", action="store_true")
        train_parser.add_argument("--epochs")

        args = parser.parse_args()
        return args
