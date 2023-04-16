import csv
import itertools
from typing import List
import networkx
from networkx.algorithms.community import k_clique_communities


INPUT_ROWS = 10000


class Cast:
    def __init__(self, csv_row):
        self.movie_id = csv_row[0]
        self.movie_title = csv_row[1]
        self.actor_name = csv_row[2]
        self.role_type = csv_row[3]
        roles = csv_row[4].split(':', 1)
        role_prefix = roles[0]
        try:
            role_content = roles[1]
        except IndexError:
            role_content = ""
        self.role_prefix = role_prefix
        self.role_content = role_content


def load_data(file_name):
    data = []
    with open(file_name) as csv_file:
        csv_reader = itertools.islice(csv.reader(csv_file, delimiter=';'), INPUT_ROWS)
        [data.append(row) for row in csv_reader]
    return data


def build_graph(records: List) -> networkx.Graph:
    movie_actors = {}  # {movie: [actor_1, actor_2, ...]}
    graph = networkx.Graph()
    for record in records:
        graph.add_node(record.actor_name)
        if record.movie_title not in movie_actors:
            movie_actors[record.movie_title] = []
        movie_actors[record.movie_title].append(record.actor_name)

    for actor_list in movie_actors.values():
        for actor_pair in list(itertools.combinations(actor_list, 2)):
            graph.add_edge(actor_pair[0], actor_pair[1])
    return graph


def print_centralities(graph: networkx.Graph):
    centralities = [
        networkx.degree_centrality,
        networkx.closeness_centrality,
        networkx.betweenness_centrality,
    ]

    print("="*14 + "Centralities" + "="*14)
    for centrality in centralities:
        centrality_result = centrality(graph)
        # Add as node attribute
        for actor, centrality_value in centrality_result.items():
            graph._node[actor][centrality.__name__] = centrality_value
        centrality_res_sorted = sorted(centrality_result.items(), key=lambda element: element[1], reverse=True)
        print(f"{centrality.__name__} - Top 5: ")
        for element in centrality_res_sorted[:5]:
            print(f"  {element[0]} ({element[1]})")
    print("=" * 40)


def print_communities(graph: networkx.Graph):
    communities = {node: cid + 1 for cid, community in enumerate(k_clique_communities(graph, 3))
                   for node in community}

    # Group actors from same communities together
    actor_communities = {}
    for key, val in communities.items():
        if val not in actor_communities:
            actor_communities[val] = []
        actor_communities[val].append(key)

    # Sort based on the length of the list of actors
    sorted_actor_communities = sorted(actor_communities.items(), key=lambda element: len(element[1]), reverse=True)

    print("="*7 + "Top 5 biggest communities" + "="*8)
    for community in sorted_actor_communities[:5]:
        print(f"ID {community[0]}, {len(community[1])} actors:")
        print(", ".join(community[1]))
        if community != sorted_actor_communities[4]:
            print("-"*40)
    print("="*40)

    for actor, community_id in communities.items():
        graph._node[actor]['community_id'] = community_id


def describe_kevin_bacon(graph, person):
    lengths = networkx.single_source_shortest_path_length(graph, person)

    # Add as node attribute
    for actor in graph.nodes():
        graph._node[actor]['KevinBacon'] = -1

    length_sum = 0
    length_count = 0
    for actor, length in lengths.items():
        graph._node[actor]['KevinBacon'] = length
        length_sum += length
        length_count += 1

    bacon_average = length_sum / length_count

    lengths_sorted = sorted(lengths.items(), key=lambda element: element[1], reverse=True)
    print("="*10 + "Kevin Bacon Numbers" + "="*11)
    print(f"From person: {person}")
    print(f"Average: {bacon_average}")
    print(f"Nearest 5:")
    nearest = lengths_sorted[-5:]
    nearest.reverse()
    for actor in nearest:
        print(f"  {actor[0]} ({actor[1]})")
    print(f"Furthest 5:")
    for actor in lengths_sorted[:5]:
        print(f"  {actor[0]} ({actor[1]})")
    print("=" * 40)


if __name__ == '__main__':
    casts = load_data('casts.csv')
    records = []
    for row in casts:
        if row[2] != "" and not row[2].startswith("s a") and not row[2].startswith("sa"):
            records.append(Cast(row))
    print("=" * 17 + "Input" + "=" * 18)
    print(f"Loaded {len(records)} actors")
    print("=" * 40)
    graph = build_graph(records)
    # Provide general statistics about the dataset
    print("="*17 + "Stats" + "="*18 )
    print(f"Number of nodes:      {graph.number_of_nodes()}")
    print(f"Number of edges:      {graph.number_of_edges()}")
    print(f"Density:              {networkx.density(graph)}")
    print(f"Number of conponents: {networkx.number_connected_components(graph)}")
    print("=" * 40)
    # Provide list of top key players using different centralities
    print_centralities(graph)
    # Describe top clusters/communities
    print_communities(graph)
    # Describe „Kevin Bacon“ numbers
    # - e.g. top actors with the highest/lowest number
    # - average number
    describe_kevin_bacon(graph, 'Tom Hanks')
    networkx.write_gexf(graph, '../results/graph.gexf')
