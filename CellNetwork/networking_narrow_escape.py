




particles = 100
D = 300 
tt = 100 # seconds for example
r = 25  # radius of sphere (i.e. cell) given in um 
ep = 10  # Radius of each, given in um 

# Calculating neighbours is intensive, lets do it once just. 
nbrs = {n[0]:[ns for ns in G.neighbors(n[0])] for n in G.nodes(data=True)}
weights = G.weights_to_A()

Cn = np.zeros(G.number_of_nodes())

for cell in G.nodes(data=True):
    for p0 in range(int(cell[1]['C']*particles)):
        cur_pos = cell[0]
        te = 0
        while te < tt:
            N = len(nbrs[cur_pos])
            # calculate escape time
            # Assume each PD field is equal ~
            w = weights[cur_pos][nbrs[cur_pos]]
            cur_pos = np.random.choice(nbrs[cur_pos], p=w/w.sum())
            te += narrow_escape(r, D, N, ep)
        Cn[cur_pos] += 1    
