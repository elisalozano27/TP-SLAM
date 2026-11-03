# segments.py - Extraction et manipulation de segments à partir d'un scan du LD06
#
# Un point    = (x, y) en mètres
# Un segment  = (A, B) avec A et B deux points
#
# Fonctions :
#   nuage_vers_segments(points)   nuage de points -> liste de segments
#   ajuster_segment(points)       régression linéaire : meilleur segment passant par des points
#   angle_entre(s1, s2)           angle entre deux segments (0 à 90°)
#   sont_proches(s1, s2)          proches en angle ET en position ?
#   fusionner(s1, s2)             un seul segment à partir de deux segments proches
#   intersection(s1, s2)          point d'intersection, ou None
#   decouper(s, coupant)          coupe s par coupant et garde le plus long côté
#
# Lancer : python segments.py        -> segments extraits de scan_01.csv
#          python segments.py test   -> tests des fonctions et des cas limites

import math

TOL = 1e-6           # tolérance numérique (m) : deux longueurs qui diffèrent de moins de TOL sont égales

# Réglages de l'extraction
SAUT = 0.15          # écart entre deux points voisins au-delà duquel on change d'objet (m)...
SAUT_PAR_M = 0.05    # ... + ce supplément par mètre de distance au robot (les points s'écartent au loin)
SEUIL_SPLIT = 0.05   # écart maximal d'un point à son segment (m)
MIN_POINTS = 5       # nombre minimum de points pour faire un segment
MORCEAU_MIN = 0.10   # longueur minimale d'un morceau avant fusion (m)
LONGUEUR_MIN = 0.40  # longueur minimale d'un segment après fusion (m)

# Réglages de "proches"
ANGLE_MAX = 10.0     # écart d'angle maximal (°)
ECART_MAX = 0.08     # écart maximal sur le côté (perpendiculaire au mur) (m)
TROU_MAX = 0.70      # trou maximal dans l'axe du mur entre les deux segments (m)


# ---------- Outils de base ----------
def longueur(s):
    (ax, ay), (bx, by) = s
    return math.hypot(bx - ax, by - ay)


def distance_point_segment(p, s):
    """Distance la plus courte entre le point p et le segment s."""
    (px, py), ((ax, ay), (bx, by)) = p, s
    ux, uy = bx - ax, by - ay
    l2 = ux * ux + uy * uy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * ux + (py - ay) * uy) / l2))
    return math.hypot(px - (ax + t * ux), py - (ay + t * uy))


# ---------- 1. Nuage de points -> segments ----------
def couper_en_groupes(points, origine=(0, 0)):
    """Sépare les points en groupes : on change de groupe quand deux voisins sont trop éloignés.
    Le seuil grandit avec la distance au robot, car les points lointains sont plus espacés."""
    groupes = [[points[0]]]
    for p, q in zip(points, points[1:]):
        if math.dist(p, q) > SAUT + SAUT_PAR_M * math.dist(origine, q):
            groupes.append([])
        groupes[-1].append(q)
    return groupes


def ajuster_segment(points):
    """Régression linéaire : droite qui passe au plus près de tous les points (moindres carrés),
    limitée par les projections du premier et du dernier point sur cette droite."""
    n = len(points)
    mx = sum(x for x, _ in points) / n                     # centre du nuage
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    syy = sum((y - my) ** 2 for _, y in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    a = 0.5 * math.atan2(2 * sxy, sxx - syy)               # direction de la droite
    ux, uy = math.cos(a), math.sin(a)

    def projeter(p):
        t = (p[0] - mx) * ux + (p[1] - my) * uy
        return mx + t * ux, my + t * uy

    return projeter(points[0]), projeter(points[-1])


def split(points):
    """Split : si un point est trop loin du segment premier -> dernier, on coupe en deux à ce point.
    Sinon, on ajuste un segment sur tous les points du groupe."""
    s = (points[0], points[-1])
    ecarts = [distance_point_segment(p, s) for p in points]
    i = ecarts.index(max(ecarts))
    if ecarts[i] > SEUIL_SPLIT:
        return split(points[:i + 1]) + split(points[i:])
    if len(points) >= MIN_POINTS and longueur(s) >= MORCEAU_MIN:
        return [ajuster_segment(points)]
    return []                                    # trop petit : bruit, on l'ignore


def nuage_vers_segments(points, origine=(0, 0)):
    """Points triés dans l'ordre du balayage -> liste de segments. origine = position du robot."""
    segments = []
    for groupe in couper_en_groupes(points, origine):
        segments += split(groupe)
    # Un même mur peut avoir été coupé en morceaux (point aberrant, passage à 0°) :
    # on fusionne les segments proches jusqu'à ce qu'il n'y en ait plus
    i = 0
    while i < len(segments):
        for j in range(i + 1, len(segments)):
            if sont_proches(segments[i], segments[j]):
                segments[i] = fusionner(segments[i], segments.pop(j))
                break                            # segments[i] a changé : on recommence ses tests
        else:
            i += 1                               # plus rien à fusionner avec segments[i]
    # Seulement maintenant, on retire les segments trop courts (morceaux restés seuls)
    return [s for s in segments if longueur(s) >= LONGUEUR_MIN]


# ---------- 2. Angle entre deux segments ----------
def angle_entre(s1, s2):
    """Angle entre les deux droites, entre 0° et 90° (le sens des segments ne compte pas)."""
    (ax, ay), (bx, by) = s1
    (cx, cy), (dx, dy) = s2
    ux, uy, vx, vy = bx - ax, by - ay, dx - cx, dy - cy
    a = abs(math.degrees(math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)))   # 0 à 180°
    return min(a, 180 - a)


# ---------- 3. Segments proches ----------
def sont_proches(s1, s2):
    """Proches = presque la même direction, presque sur la même droite (écart sur le côté),
    et pas trop loin l'un de l'autre le long de cette droite (trou dans l'axe)."""
    if angle_entre(s1, s2) > ANGLE_MAX:
        return False
    long_, court = (s1, s2) if longueur(s1) >= longueur(s2) else (s2, s1)
    (ax, ay), (bx, by) = long_
    l = longueur(long_)
    ux, uy = (bx - ax) / l, (by - ay) / l            # direction du plus long
    # position des extrémités du plus court : t le long du plus long, n sur le côté
    t = [(px - ax) * ux + (py - ay) * uy for px, py in court]
    n = [abs(-(px - ax) * uy + (py - ay) * ux) for px, py in court]
    trou = max(0.0, min(t) - l, 0.0 - max(t))        # 0 si les deux segments se chevauchent
    return max(n) <= ECART_MAX and trou <= TROU_MAX


# ---------- 4. Fusion ----------
def fusionner(s1, s2):
    """Fusionne deux segments proches : on garde la droite du plus long
    et on l'allonge pour couvrir les deux segments."""
    long_ = s1 if longueur(s1) >= longueur(s2) else s2
    (ax, ay), (bx, by) = long_
    l = longueur(long_)
    ux, uy = (bx - ax) / l, (by - ay) / l
    # position de chacune des 4 extrémités le long de cette droite
    ts = [(px - ax) * ux + (py - ay) * uy for (px, py) in (*s1, *s2)]
    return (ax + min(ts) * ux, ay + min(ts) * uy), (ax + max(ts) * ux, ay + max(ts) * uy)


# ---------- 5. Intersection ----------
def parametres(s1, s2):
    """Cherche t et u tels que  A + t(B - A) = C + u(D - C).
    t = 0 en A, t = 1 en B (idem pour u sur CD). Renvoie None si les droites sont parallèles."""
    (ax, ay), (bx, by) = s1
    (cx, cy), (dx, dy) = s2
    rx, ry, sx, sy = bx - ax, by - ay, dx - cx, dy - cy
    den = rx * sy - ry * sx                      # produit vectoriel : 0 si parallèles
    if abs(den) <= TOL * math.hypot(rx, ry):     # parallèles ou colinéaires (à TOL près)
        return None
    wx, wy = cx - ax, cy - ay
    return (wx * sy - wy * sx) / den, (wx * ry - wy * rx) / den


def intersection(s1, s2):
    """Point commun aux deux segments (extrémités comprises), ou None.
    Colinéaires : None (il n'y a pas un seul point d'intersection)."""
    p = parametres(s1, s2)
    if p is None:
        return None
    t, u = p
    l1, l2 = longueur(s1), longueur(s2)
    if -TOL <= t * l1 <= l1 + TOL and -TOL <= u * l2 <= l2 + TOL:
        (ax, ay), (bx, by) = s1
        return ax + t * (bx - ax), ay + t * (by - ay)
    return None


# ---------- 6. Découpage ----------
def decouper(s, coupant):
    """Coupe s là où coupant le traverse et garde le plus long côté.
    s est renvoyé inchangé si : parallèles/colinéaires, intersection hors de coupant,
    ou intersection à une extrémité de s (pas strictement à l'intérieur)."""
    p = parametres(s, coupant)
    if p is None:
        return s
    t, u = p
    l, lc = longueur(s), longueur(coupant)
    sur_coupant = -TOL <= u * lc <= lc + TOL     # sur coupant, extrémités comprises
    dans_s = TOL < t * l < l - TOL               # strictement à l'intérieur de s
    if not (sur_coupant and dans_s):
        return s
    (ax, ay), (bx, by) = A, B = s
    i = (ax + t * (bx - ax), ay + t * (by - ay))
    cote_a, cote_b = t * l, (1 - t) * l          # longueur côté A et côté B
    return (A, i) if cote_a >= cote_b - TOL else (i, B)   # égalité -> côté du premier point A


# ---------- Tests (lancer : python segments.py test) ----------
def tests():
    def egal(a, b):
        """Compare deux points ou deux segments à 1e-9 près."""
        if isinstance(a[0], tuple):
            return all(egal(p, q) for p, q in zip(a, b))
        return math.dist(a, b) < 1e-9

    def test(nom, condition):
        print(("OK    " if condition else "ECHEC ") + nom)

    # Angle
    test("angle : perpendiculaires = 90°", abs(angle_entre(((0, 0), (1, 0)), ((0, 0), (0, 1))) - 90) < 1e-9)
    test("angle : sens opposés = 0°", abs(angle_entre(((0, 0), (1, 0)), ((5, 1), (3, 1)))) < 1e-9)
    test("angle : 45°", abs(angle_entre(((0, 0), (1, 0)), ((0, 0), (1, 1))) - 45) < 1e-9)

    # Proches
    mur = ((0, 0), (1, 0))
    test("proches : parallèle à 5 cm", sont_proches(mur, ((0.5, 0.05), (1.5, 0.05))))
    test("pas proches : parallèle à 1 m", not sont_proches(mur, ((0, 1), (1, 1))))
    test("pas proches : alignés mais 1 m d'écart", not sont_proches(mur, ((2, 0), (3, 0))))
    test("pas proches : se touchent mais 30°", not sont_proches(mur, ((1, 0), (2, math.tan(math.radians(30))))))
    test("proches : alignés avec 30 cm de trou", sont_proches(mur, ((1.3, 0), (2, 0))))
    test("pas proches : parallèles à 15 cm (deux murs)", not sont_proches(mur, ((0, 0.15), (1, 0.15))))

    # Fusion
    test("fusion : couvre les deux segments", egal(fusionner(mur, ((0.8, 0), (2, 0))), ((0, 0), (2, 0))))

    # Intersection
    test("intersection : croix", egal(intersection(((0, 0), (1, 1)), ((0, 1), (1, 0))), (0.5, 0.5)))
    test("intersection : extrémité commune", egal(intersection(mur, ((1, 0), (1, 1))), (1, 0)))
    test("intersection : hors des segments", intersection(mur, ((2, -1), (2, 1))) is None)
    test("intersection : parallèles", intersection(mur, ((0, 1), (1, 1))) is None)
    test("intersection : colinéaires", intersection(mur, ((0.5, 0), (2, 0))) is None)

    # Découpage (s va de A = (0, 0) à B = (3, 0))
    s = ((0, 0), (3, 0))
    test("découpe : traversé à x = 1 -> garde le côté B", egal(decouper(s, ((1, -1), (1, 1))), ((1, 0), (3, 0))))
    test("découpe : traversé à x = 2 -> garde le côté A", egal(decouper(s, ((2, -1), (2, 1))), ((0, 0), (2, 0))))
    test("pas de découpe : intersection à l'extrémité B", decouper(s, ((3, -1), (3, 1))) == s)
    test("pas de découpe : intersection à l'extrémité A", decouper(s, ((0, -1), (0, 1))) == s)
    test("découpe : coupant touche s par son extrémité (T)", egal(decouper(s, ((1, 0), (1, 1))), ((1, 0), (3, 0))))
    test("pas de découpe : intersection hors du coupant", decouper(s, ((1, 0.5), (1, 1))) == s)
    test("pas de découpe : colinéaires", decouper(s, ((1, 0), (2, 0))) == s)
    test("pas de découpe : parallèles", decouper(s, ((1, 1), (2, 1))) == s)

    # Cas limites avec la tolérance
    s2 = ((0, 0), (2, 0))
    test("égalité exacte -> garde le côté du premier point A", egal(decouper(s2, ((1, -1), (1, 1))), ((0, 0), (1, 0))))
    x = 1 - TOL / 4    # côté A plus court, mais de moins de TOL
    test("égalité à TOL près -> garde le côté de A", egal(decouper(s2, ((x, -1), (x, 1))), ((0, 0), (x, 0))))
    x = 1 - 10 * TOL   # côté A plus court de plus de TOL
    test("différence > TOL -> garde le côté B", egal(decouper(s2, ((x, -1), (x, 1))), ((x, 0), (2, 0))))
    x = 3 - TOL / 2    # à moins de TOL de l'extrémité B
    test("à TOL près de l'extrémité -> pas de découpe", decouper(s, ((x, -1), (x, 1))) == s)
    x = 3 - 10 * TOL   # un peu plus loin que TOL de l'extrémité B
    test("juste après TOL de l'extrémité -> découpe", egal(decouper(s, ((x, -1), (x, 1))), ((0, 0), (x, 0))))
    test("coupant s'arrête à TOL/2 de s -> compte comme touchant", egal(decouper(s, ((1, TOL / 2), (1, 1))), ((1, 0), (3, 0))))
    test("coupant s'arrête à 10·TOL de s -> pas de découpe", decouper(s, ((1, 10 * TOL), (1, 1))) == s)

    # Nuage de points -> segments : robot au centre d'une pièce carrée de 4 m, un point par degré
    carre = []
    for a in range(360):
        t = math.radians(a)
        r = 2 / max(abs(math.cos(t)), abs(math.sin(t)))
        carre.append((r * math.cos(t), r * math.sin(t)))
    segs = nuage_vers_segments(carre)
    test(f"pièce carrée -> 4 segments (trouvé : {len(segs)})", len(segs) == 4)


# ---------- Exemple sur un vrai scan (lancer : python segments.py) ----------
def afficher_scan(fichier):
    import matplotlib.pyplot as plt
    from grille import lire_scan

    scan = sorted(lire_scan(fichier))                        # trié par angle = ordre du balayage
    points = [(r * math.cos(a), r * math.sin(a)) for a, r in scan]
    segs = nuage_vers_segments(points)
    print(len(segs), "segments trouvés")

    plt.figure(figsize=(7, 7))
    plt.scatter(*zip(*points), s=3, color="lightgray", label="points")
    for (ax, ay), (bx, by) in segs:
        plt.plot([ax, bx], [ay, by], "-o", linewidth=2, markersize=3)
    plt.plot(0, 0, "r^", label="robot")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"{len(segs)} segments extraits de {fichier}")
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        tests()
    else:
        afficher_scan("scan_01.csv")