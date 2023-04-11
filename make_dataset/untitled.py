def gap_lcs(lh, rh): 
    ln, rn = len(lh), len(rh) 
    k1 = ln
    k2 = rn

    memo = {}
    choices = {}
    def rec(li, ri, l_budget, r_budget):
        # At the end of either sequence we are forced to use
        # Case 1: terminate the match.
        if li >= ln or ri >= rn:
            return 0 

        # Cache results. This limits the complexity to O(ln * lm * k^2).
        # Without this the recursion would take exponential time.
        key = (li, ri, l_budget, r_budget) 
        if key in memo: 
            return memo[key]

        # Case 1: terminate the match.
        res = 0
        choice = (0, 0)

        # Case 2: matching characters, extend the sequence.
        if lh[li] == rh[ri]:
            test = 1 + rec(li + 1, ri + 1, k1, k2)
            if test > res:
                res = test
                choice = (1, 1)

        # Case 3: skip the left character if there's still budget.
        if l_budget > 0:
            test = rec(li + 1, ri, l_budget - 1, r_budget)
            if test > res:
                res = test
                choice = (1, 0)

        # Case 4: skip the right character if there's still budget.
        if r_budget > 0:
            test = rec(li, ri + 1, l_budget, r_budget - 1)
            if test > res:
                res = test
                choice = (0, 1)

        memo[key] = res
        choices[key] = choice
        return res

    # Find the best combination of starting points within the two sequences.
    # This is so the gap constraint will not apply to skips at the start.
    res = 0
    best_li, best_ri = 0, 0
    for li in range(ln):
        for ri in range(rn):
            test = rec(li, ri, k1, k2)
            if test > res:
                res, best_li, best_ri = test, li, ri

    # Reconstruct the LCS by following the choices we tracked,
    # starting from the best start we found.
    li, ri = best_li, best_ri
    l_budget, r_budget = k1, k2

    path = []
    while True:
        key = (li, ri, l_budget, r_budget)

        # Case 1.
        if key not in choices:
            break

        inc_li, inc_ri = choices[key]

        # Case 1.
        if inc_li == 0 and inc_ri == 0:
            break

        if inc_li == 1 and inc_ri == 1:
            # Case 2.
            l_budget, r_budget = k1, k2
            path.append((lh[li], li, ri))
        else:
            # Cases 3 and 4.
            l_budget -= inc_li
            r_budget -= inc_ri

        li += inc_li
        ri += inc_ri

    return path



X = "AGLPGTHXAB"
Y = "GGXTXAYB"


print(f"Length of LCS is {gap_lcs(X, Y)} = {len(gap_lcs(X, Y))}")

# This code is contributed by shinjanpatra

