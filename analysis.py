import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["Bacteria","Plant", "Protist", "Cnidaria", "Fungi","Animalia",
                "Invertebrate", "Vertebrate", 
                "Amphibia", "Fish", "Mammal", "Bird", "Reptile", "Ecdysozoa", "Spiralia",
                "HumanHost"]

def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2g}',  # shows up to 2 significant digits, trims trailing zeros
                    (p.get_x() + p.get_width() / 2., height + 0.05 * height),
                    ha='center', va='bottom', fontweight="bold",
                    fontsize=10, rotation=90)

def custom_palette():

    # Define label groups and assign discrete viridis tones
    label_groups = {
        "Group1": ["Bacteria", "Plant", "Protist", "Cnidaria", "Fungi", "Animalia"],
        "Group2": ["Invertebrate", "Vertebrate"],
        "Group3": ["Amphibia", "Fish", "Mammal", "Bird", "Reptile"],
        "Group4": ["Ecdysozoa", "Spiralia"],
        "Group5": ["HumanHost"]
    }

    # Flatten labels and assign group colors
    group_palette = sns.color_palette("viridis", 5)
    group_color_map = dict(zip(label_groups.keys(), group_palette))

    # Create label-to-color mapping
    label_color_map = {}
    for group, group_labels in label_groups.items():
        for label in group_labels:
            label_color_map[label] = group_color_map[group]

    # Final palette for plotting
    custom_palette = [label_color_map[label] for label in labels]
    return custom_palette

def draw_custom_vertical_lines(ax, label_order):
    boundaries = [("Animalia", "Invertebrate"),
                  ("Vertebrate", "Amphibia"),
                  ("Spiralia", "HumanHost")]
    for left, right in boundaries:
        if left in label_order and right in label_order:
            left_idx = label_order.index(left)
            right_idx = label_order.index(right)
            x_pos = (left_idx + right_idx) / 2
            ax.axvline(x=x_pos, color='black', linestyle='--', linewidth=1)


def get_labels(df, suffix):
    label_sums = df[labels].sum()
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=label_sums.index, y=label_sums.values, palette=custom_palette())
    annotate_bars(ax)
    draw_custom_vertical_lines(ax, labels)
    plt.xticks(rotation=45, fontsize=12)
    plt.title(f'Total Sequences with Label - {suffix}')
    plt.ylabel('Sequences')
    plt.xlabel("Labels")
    plt.tight_layout()
    plt.savefig(f'test/analysis/label_sums_{suffix}.png')
    plt.close()

def get_partitions(df, suffix):
    partition_counts = df["partition"].value_counts()
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=partition_counts.index, y=partition_counts.values, palette="magma")
    annotate_bars(ax)
    plt.title(f'Sequences per Partition - {suffix}')
    plt.ylabel('Sequences')
    plt.xlabel('Partition')
    plt.tight_layout()
    plt.savefig(f'test/analysis/partition_counts_{suffix}.png')
    plt.close()

def get_partitions_clusters(df, cluster, suffix):
    partitions = df["partition"].unique()
    cluster_counts = {p: df[df["partition"] == p][cluster].nunique() for p in partitions}
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=list(cluster_counts.keys()), y=list(cluster_counts.values()), palette="coolwarm")
    annotate_bars(ax)
    plt.title(f'Unique Clusters per Partition - {suffix}')
    plt.ylabel(f'Clusters')
    plt.xlabel('Partition')
    plt.tight_layout()
    plt.savefig(f'test/analysis/unique_clusters_{suffix}.png')
    plt.close()

def get_partition_labels(df, suffix):
    partitions = df["partition"].unique()
    num_partitions = len(partitions)
    fig, axes = plt.subplots(num_partitions, 1, figsize=(14, 5 * num_partitions), constrained_layout=True)

    if num_partitions == 1:
        axes = [axes]

    for ax, p in zip(axes, partitions):
        label_sums = df[df["partition"] == p][labels].sum()
        sns.barplot(x=label_sums.index, y=label_sums.values, palette=custom_palette(), ax=ax)
        annotate_bars(ax)
        draw_custom_vertical_lines(ax, labels)
        ax.set_title(f'Sequences per Label for Partition {p} - {suffix}')
        ax.set_ylabel('Sequences')
        ax.set_xlabel('Labels')
        ax.set_xticklabels(label_sums.index, rotation=45, fontsize=12)

    fig.savefig(f'test/analysis/label_sums_all_partitions_{suffix}.png')
    plt.close()

def get_partitioncluster_labels(df, cluster, suffix):
    partitions = df["partition"].unique()
    num_partitions = len(partitions)
    fig, axes = plt.subplots(num_partitions, 1, figsize=(14, 5 * num_partitions), constrained_layout=True)

    if num_partitions == 1:
        axes = [axes]

    for ax, p in zip(axes, partitions):
        partition_df = df[df["partition"] == p]
        normalized_label_frequencies = partition_df.groupby(cluster)[labels].mean()
        label_sums = normalized_label_frequencies.sum()
        sns.barplot(x=label_sums.index, y=label_sums.values, palette=custom_palette(), ax=ax)
        annotate_bars(ax)
        draw_custom_vertical_lines(ax, labels)
        ax.set_title(f'Clusters per Label for Partition {p} - {suffix}')
        ax.set_ylabel('Representatives')
        ax.set_xlabel('Labels')
        ax.set_xticklabels(label_sums.index, rotation=45, fontsize=12)

    fig.savefig(f'test/analysis/label_clustersums_all_partitions_{suffix}.png')
    plt.close()

def influenza_clusters(df, cluster, suffix):
    partitions = df["partition"].unique()
    plot_data = []

    for p in partitions:
        df_partition = df[df["partition"] == p]
        df_influenza = df_partition[df_partition["Virus Lineage"].str.contains("nfluenza", case=False, na=False)]
        df_non_influenza = df_partition[~df_partition["Virus Lineage"].str.contains("nfluenza", case=False, na=False)]

        total_clusters = df_partition[cluster].nunique()
        influenza_clusters = df_influenza[cluster].nunique()
        non_influenza_clusters = df_non_influenza[cluster].nunique()

        plot_data.append({"Partition": p, "Category": "Total", "Clusters": total_clusters})
        plot_data.append({"Partition": p, "Category": "Influenza", "Clusters": influenza_clusters})
        plot_data.append({"Partition": p, "Category": "Non-Influenza", "Clusters": non_influenza_clusters})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Partition", y="Clusters", hue="Category", palette="coolwarm")
    annotate_bars(ax)
    plt.title(f'Unique Clusters per Partition - Influenza Split - {suffix}')
    plt.ylabel('Clusters')
    plt.xlabel('Partition')
    plt.legend(title="Category")
    plt.tight_layout()
    plt.savefig(f'test/analysis/unique_clusters_influenza_split_{suffix}.png')
    plt.close()



def influenza_sequences(df, suffix):
    partitions = df["partition"].unique()
    plot_data = []

    for p in partitions:
        df_partition = df[df["partition"] == p]
        df_influenza = df_partition[df_partition["Virus Lineage"].str.contains("nfluenza", case=False, na=False)]
        df_non_influenza = df_partition[~df_partition["Virus Lineage"].str.contains("nfluenza", case=False, na=False)]

        total_sequences = len(df_partition)
        influenza_sequences = len(df_influenza)
        non_influenza_sequences = len(df_non_influenza)

        plot_data.append({"Partition": p, "Category": "Total", "Sequences": total_sequences})
        plot_data.append({"Partition": p, "Category": "Influenza", "Sequences": influenza_sequences})
        plot_data.append({"Partition": p, "Category": "Non-Influenza", "Sequences": non_influenza_sequences})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Partition", y="Sequences", hue="Category", palette="magma")
    annotate_bars(ax)
    plt.title(f'Sequences per Partition - Influenza Split - {suffix}')
    plt.ylabel('Sequences')
    plt.xlabel('Partition')
    plt.legend(title="Category")
    plt.tight_layout()
    plt.savefig(f'test/analysis/partition_counts_influenza_split_{suffix}.png')
    plt.close()


## WHICH PARTITIONS HAVE INFLUENZA


reduxdf = pd.read_csv("/work3/alff/ViralInf/RNA_db/data/metadata/repr_dbredux95_origlabels_partition.tsv", sep="\t")
get_labels(reduxdf, "redux95")
get_partitions(reduxdf, "redux95")
get_partition_labels(reduxdf, "redux95")
influenza_sequences(reduxdf, "redux95")
exit()
reduxdf = pd.read_csv("/work3/alff/ViralInf/RNA_db/data/metadata/repr_db_origlables_partition045.tsv", sep="\t")
get_labels(reduxdf, "cluster045")
get_partitions(reduxdf, "cluster045")
get_partition_labels(reduxdf,  suffix="cluster045")
get_partitions_clusters(reduxdf, cluster="Cluster0.45", suffix="cluster045")
get_partitioncluster_labels(reduxdf, cluster="Cluster0.45", suffix="cluster045")
influenza_sequences(reduxdf, "cluster045")
influenza_clusters(reduxdf, cluster="Cluster0.45",  suffix="cluster045")
reduxdf = pd.read_csv("/work3/alff/ViralInf/RNA_db/data/metadata/repr_db_origlables_partition04.tsv", sep="\t")
get_labels(reduxdf, "cluster04")
get_partitions(reduxdf, "cluster04")
get_partition_labels(reduxdf,  suffix="cluster04")
get_partitions_clusters(reduxdf, cluster="Cluster0.4", suffix="cluster04")
get_partitioncluster_labels(reduxdf, cluster="Cluster0.4", suffix="cluster04")
influenza_sequences(reduxdf, "cluster04")
influenza_clusters(reduxdf, cluster="Cluster0.4",  suffix="cluster04")
