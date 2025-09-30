
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import numpy as np


from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import os
from matplotlib.pyplot import get_cmap
from matplotlib.colors import LinearSegmentedColormap, to_hex

os.environ["QT_QPA_PLATFORM"] = "offscreen"


class TreeOutput:
    hierarchy_simple = [
        ("root", "Bacteria"),
        ("root", "Plant"),
        ("root", "Animalia"),
        ("root", "Protist"),
        ("root", "Cnidaria"),
        ("root", "Fungi"),
        ("Animalia", "Vertebrate"),
        ("Animalia", "Invertebrate"),
        ("Vertebrate", "Amphibia"),
        ("Vertebrate", "Fish"),
        ("Vertebrate", "Mammal"),
        ("Vertebrate", "Bird"),
        ("Vertebrate", "Reptile"),
        ("Invertebrate", "Spiralia"),
        ("Invertebrate", "Ecdysozoa"),
        ("Mammal", "Human"),
    ]

    def __init__(self, hierarchy_type="simple"):
        self.node_map = {}
        self.tree = Tree(name="root")
        self.node_map["root"] = self.tree

        if hierarchy_type == "simple":
            for parent, child in TreeOutput.hierarchy_simple:
                parent_node = self.node_map[parent]
                child_node = parent_node.add_child(name=child)
                self.node_map[child] = child_node

    def fillpred_tree(self, output_vector, label_map):
        for i, value in enumerate(output_vector):
            label = label_map[i]
            if label in self.node_map:
                mean, std = value if isinstance(value, tuple) else (value, None)
                self.node_map[label].output = mean
                self.node_map[label].std = std

        for node in self.node_map.values():
            if not hasattr(node, "output"):
                node.output = None
                node.std = None

    def makepred_tree(self, output_vector, label_map, threshold):
        predicted_labels = [label_map[i] for i, score in enumerate(output_vector) if score >= threshold]

        for node in self.node_map.values():
            node.predicted = False

        for label in predicted_labels:
            if label in self.node_map:
                node = self.node_map[label]
                node.predicted = True
                for ancestor in node.get_ancestors():
                    ancestor.predicted = True

    def white_to_ylorbr_color(self, value):
        value = max(0.0, min(1.0, value))
        base_cmap = get_cmap('YlOrBr')
        colors = ['#ffffff'] + [to_hex(base_cmap(i)) for i in [0.0, 0.25, 0.5, 0.75, 1.0]]
        custom_cmap = LinearSegmentedColormap.from_list("white_to_ylorbr", colors)
        return to_hex(custom_cmap(value))



    def save_tree(self, outpath):
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.mode = "r"  # Rectangular layout

        def layout(node):
            name = node.name
            output = getattr(node, "output", None)
            std = getattr(node, "std", None)

            # Add name with larger font
            name_face = TextFace(name, fsize=12, bold=True)
            node.add_face(name_face, column=0, position="branch-right")

            # Add mean and std below the name with smaller font
            if output is not None or std is not None:
                stats = []
                if output is not None:
                    stats.append(f"μ={output:.2f}")
                if std is not None:
                    stats.append(f"σ={std:.2f}")
                stats_face = TextFace("\n"+", ".join(stats), fsize=9)
                node.add_face(stats_face, column=0, position="branch-right")

            nstyle = NodeStyle()
            nstyle["shape"] = "circle"
            nstyle["fgcolor"] = "black"

            # Encode mean as node size (scaled between 5 and 20)
            if output is not None:
                output_clamped = max(0.0, min(1.0, output))
                nstyle["size"] = int(5 + 15 * output_clamped)
                nstyle["bgcolor"] = self.white_to_ylorbr_color(output)
            else:
                nstyle["size"] = 10

            node.set_style(nstyle)

        ts.layout_fn = layout
        self.tree.render(f"{outpath}.png", tree_style=ts)






# Example usage
if __name__ == '__main__':
    tree = TreeOutput(hierarchy_type="simple")
    output_vector = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            [0.3, 0.2, 0.4, 0.4, 0.5, 0.1, 0.2],
                            [0.2, 0.2, 0.6, 0.4, 0.5, 0.6, 0.7]])
    mean_vector = np.mean(output_vector, axis=0)
    std_vector = np.std(output_vector, axis=0)
    print(mean_vector, std_vector)
    output_vector = []
    for mean, std in zip(mean_vector, std_vector):
        output_vector.append((mean,std))
    label_map = ["Bacteria", "Plant", "Animalia", "Vertebrate", "Mammal", "Human", "Fish"]
    tree.fillpred_tree(output_vector, label_map)
    tree.makepred_tree([x[0] for x in output_vector], label_map, threshold=0.5)
    tree.save_tree("hierarchical_tree")
