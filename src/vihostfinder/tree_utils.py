import networkx as nx
import matplotlib.pyplot as plt


class TreeOutput:

    hierarchy_simple = [
                ("root", "Bacteria"),
                ("root", "Plant"),
                ("root", "Animalia"),
                ("root", "Protist"),
                ("root", "Cnidaria"),
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

        self.tree = nx.DiGraph()

        if hierarchy_type == "simple":
            self.tree.add_edges_from(TreeOutput.hierarchy_simple)
    
    def print_tree(self):
        # Print tree with output values
        for node in self.tree.nodes:
            print(f"Node: {node}, Output: {self.tree.nodes[node]['output']}")


    def fillpred_tree(self, output_vector, label_map):
        # Attach output values to nodes
        for i, value in enumerate(output_vector):
            label = label_map[i]
            if label in self.tree.nodes:
                self.tree.nodes[label]["output"] = value

        # Optional: initialize other nodes with None
        for node in self.tree.nodes:
            self.tree.nodes[node].setdefault("output", None)

        # Print tree with output values
        for node in self.tree.nodes:
            print(f"Node: {node}, Output: {self.tree.nodes[node]['output']}")

    def makepred_tree(self, output_vector, label_map, threshold):
    
        # Get predicted labels
        predicted_labels = [label_map[i] for i, score in enumerate(output_vector) if score >= threshold]

        # Mark predicted nodes in the graph
        for node in self.tree.nodes:
            self.tree.nodes[node]["predicted"] = node in predicted_labels

        # Optional: expand predictions to include ancestors
        for node in predicted_labels:
            ancestors = nx.ancestors(self.tree, node)
            for ancestor in ancestors:
                self.tree.nodes[ancestor]["predicted"] = True

        # Print tree with prediction flags
        for node in self.tree.nodes:
            print(f"Node: {node}, Predicted: {self.tree.nodes[node]['predicted']}")

    def save_tree(self, outpath):

        # Draw the tree
        pos = nx.spring_layout(self.tree)  # or use nx.nx_agraph.graphviz_layout(G) if you have pygraphviz
        labels = {node: f"{node}\n{self.tree.nodes[node]['output']}" for node in self.tree.nodes}

        plt.figure(figsize=(8, 6))
        nx.draw(self.tree, pos, with_labels=True, labels=labels, node_color="lightblue", node_size=1500, font_size=10)
        plt.title("Tree with Output Values")
        plt.tight_layout()
        plt.savefig("{}.png".format(outpath))  # Saves the image
        plt.close()



