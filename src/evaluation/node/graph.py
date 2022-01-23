import networkx as nx 
import proplot as pplt

def collect_parents(graph, labels, start):
    for slot, p in start.parents_iter:
        graph.add_edge(start.name, p.name, label=slot)
        labels[(start.name, p.name)] = slot
        collect_parents(graph, labels, p)

    return labels

def trim_pos_y(pos, factor):
    new_pos = {}
    for k, (x, y) in pos.items():
        new_pos[k] = (x, y * factor)

    return new_pos

def draw_evaluation_tree(start, out, figsize=(100, 20)):
    fig, ax = pplt.subplots(figsize=figsize, ncols=1, nrows=1)
    G = nx.DiGraph()
    labels = collect_parents(G, {}, start)
    pos = trim_pos_y(nx.nx_pydot.pydot_layout(G, prog='dot'), 1/8)
    nx.draw_networkx(G, ax=ax, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    fig.savefig(out)
