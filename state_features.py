#How many cells of that color are on the flood boundary
#How many are one layer behind 
#How many cells of each color exist outside the flood 

from flood_it import Board, Config

def feature(board : Board):
    n = board.config.size
    flooded = board.connected()
    current = board.grid[0][0]

    #color remaining
    
    #remaping flooded color to 0
    remap = {current : 0}
    for c in range(board.config.colors):
            if c not in remap:
                remap[c] = len(remap)

    #canonical id back to real color mapping
    inv = {v:k for k,v in remap.items()}


    #neigbhor function = gives th 4 in grid neighbour of a cell
    def neighbour(r,c):
        for dr,dc in ((-1,0),(1,0), (0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                yield nr,nc
    #answer if the cell is adj to cell in the region
    def touches(cell, region):
         return any(nb in region for nb in neighbour(cell[0], cell[1]))


    #the three layers
    outside = {(r,c) for r in range(n) for c in range(n) if (r,c) not in flooded}
    boundary = {c for c in outside if touches(c, flooded)}
    distance_2 = {c for c in outside - boundary if touches(c, boundary)}

    #pre color counts
    l1 = [0] * board.config.colors
    l2 = [0] * board.config.colors
    out = [0] * board.config.colors

    for (r,c) in outside:
        color_ = remap[board.grid[r][c]]
        out[color_] += 1

        if(r,c) in boundary:
            l1[color_] += 1
        elif (r,c) in distance_2:
            l2[color_] += 1 

    boundary_colors = {remap[board.grid[r][c]] for (r,c) in boundary}

    return remap, inv, boundary_colors, l1, l2, out

BINS = (0,1,2,3,4,6,8,12,16,24,32,48,64)
def bucket(v):
    for i, b in enumerate(BINS):
        if v <= b:
            return i
    return len(BINS)                        

def state_key(board, moves_left):
    n = board.config.size
    _, _, _, l1, l2, out = feature(board)

    cov_b   = min(15, round(board.coverage / (n * n) * 16))          # 0..16 ratio
    moves_b = min(15, round(moves_left / max(1, board.config.move_limit) * 12))
    nbc = sum(1 for v in l1  if v > 0)      
    noc = sum(1 for v in out if v > 0)    

    slots = (moves_b, cov_b, nbc, noc,
             *[min(bucket(v), 15) for v in l1],
             *[min(bucket(v), 15) for v in l2],
             *[min(bucket(v), 15) for v in out])

    key = 0
    for v in slots:                         # base-16 pack; every slot < 16
        key = key * 16 + v
    return key