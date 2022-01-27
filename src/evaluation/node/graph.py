import networkx as nx 
import proplot as pplt

def collect_parents(graph, labels, start, use_id=True):
    for slot, p in start.parents_iter:
        if use_id:
            s_name = "{} {}".format(start.name, start.global_id)
            p_name = "{} {}".format(p.name, p.global_id)
        else:
            s_name = start.name
            p_name = p.name

        graph.add_edge(s_name, p_name, label=slot)
        labels[(s_name, p_name)] = slot
        collect_parents(graph, labels, p, use_id)

    return labels

def trim_pos_y(pos, factor):
    new_pos = {}
    for k, (x, y) in pos.items():
        new_pos[k] = (x, y * factor)

    return new_pos

def draw_node_chain(start, out, figsize=(100, 20), use_id=True):
    if not use_id:
        print("Warning: draw_evaluation_tree does not differentiate between different nodes of the same name if use_id is not set")
    fig, ax = pplt.subplots(figsize=figsize, ncols=1, nrows=1)
    G = nx.DiGraph()
    labels = collect_parents(G, {}, start, use_id)
    pos = trim_pos_y(nx.nx_pydot.pydot_layout(G, prog='dot'), 1/8)
    nx.draw_networkx(G, ax=ax, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    fig.savefig(out)
