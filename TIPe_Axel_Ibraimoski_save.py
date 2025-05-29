"""Module de classification d'images par analyse spectrale"""

import numpy as np
import matplotlib.pyplot as plt
    
ImagePath = str
Etiquete = str 
Distance = float
Vecteur = np.ndarray
ImageVecteur = tuple[np.ndarray, ImagePath, Etiquete]
ImagePathEtiquetee = tuple[ImagePath, Etiquete]
ImagePaths = list[ImagePath]
ImagePathsEtiquetees = list[ImagePathEtiquetee]
ImageVecteurs = list[ImageVecteur]


images = [ ("C:/Users/alban/Downloads/1748375982679.jpg", "soleil")] 
vecteur = ( )
# les listes, composées de sous listes de type homogène

L_Groupes_defaut = [[("C:/Users//image.jpg", "exemple1"), ("C:/Users//image2.jpg", "exemple1")] , # une norme assicié au groupe
                    [("C:/Users//image3.jpg", "exemple2"), ("C:/Users//image4.jpg", "exemple2")]] # une norme associée au groupe 


def convolution(vect: Vecteur, largeur = 20) -> Vecteur:
    """
    Les images récupérées peuvent ne pas être parfaites à cause de décalages dans le dispositif,
    seulement, on ne paut pas laisser un décalage de quelques pixels venir rendre les mesures inexploitables.
    """
    y = vect.shape[0]
    # paramètre à modifier si besoin 
    noyau = np.ones(y//largeur) / (y//largeur) 
    
    return np.convolve(vect, noyau, mode = 'same')


def traiter_image( chemin_etiq: ImagePathEtiquetee ) -> ImageVecteur :
    """ 
    Transforme une 
    Cette fonction récupere les image à partir d'une liste contenant les chemins,  
    et elle transforme ces images en une liste de vecteurs sur lesquels on peut appliquer nos algorithmes.
    """
    img = plt.imread(chemin_etiq[0])
    x, y, z = img.shape
    
    # on recupère la ligne du milieu de l'image en la grisant 
    x2 = x // 2
    
    vect = np.zeros(y)
    
    for i in range(10) : 
        # on recupere plusieurs lignes pour essayer de minimiser l'impact du bruit de la photo
        vect += (  0.299 * img[x2 + i,:,0] 
                 + 0.587 * img[x2 + i,:,1] 
                 + 0.114 * img[x2 + i,:,2] ) 
    
    # on normalise le vecteur 
    
    norme = vect.max()
    if norme != 0 :
        vect /= norme
    return (convolution(vect[:]),chemin_etiq[0],chemin_etiq[1])


def traiter_les_images( L_image_etiq: ImagePathsEtiquetees ) -> ImageVecteurs :
    """Applique la traitement sur une liste d'images"""
    return [traiter_image(chemin_etiq) for chemin_etiq in L_image_etiq]


def distance_e(vect1 : ImageVecteur, vect2: ImageVecteur) -> Distance  : 
    """Calcule la distance euclidienne entre deux vecteurs"""
    y1 = vect1.shape[0]
    y2 = vect2.shape[0]
    if y1 != y2 : 
        raise ValueError("vecteurs impcompatibles")
    else : 
        S = 0 
        for yi in range(y1) :
            S += float( vect1[yi] - vect2[yi]) ** 2 
    return S ** (1/2)

    
def k_plus_proches_voisins_v1( L_images : ImagePathsEtiquetees , chemin_image : ImagePath, k : int) ->  Etiquete :
    """
    cette fonction prend les chemins etiquetes, et le chemin de l'image, 
    elle traite les image, et definit leur distance ecludienne par rapport à l'image etudiee, 
    et renvoie l'etiquette la plus presente parmis ses k plus proches voisins    
    """
    vect_img = traiter_les_images([(chemin_image, "_")])[0][0]
    L_distances = []
    
    for chemin_etiq in L_images :
        # on recupere le chemin dans chemin_etiqu, et on rajoute sa distance à la liste
        i_vecteur = traiter_les_images([chemin_etiq])[0][0]
        distance = distance_e(i_vecteur, vect_img)
        
        L_distances.append((distance, chemin_etiq[1])) 
    
    L_distances.sort()
    # un dictionnaire pour stocker les occurences 
    dico_occurences = {} 
    
    for i in range(min(len(L_distances), k)) : 
        
        if L_distances[i][1] in dico_occurences : 
            dico_occurences[L_distances[i][1]] += 1 
        else : 
            dico_occurences[L_distances[i][1]] = 1
            
    # on renvoie ainsi la valeur la pllus presentes parmis les k plus proches voisins 
    return max(dico_occurences, key=dico_occurences.get)


def peser(L : list[ImagePathsEtiquetees], centres: list) -> float :
    """ Renvoie le poid d'une manière de mettre en cluster"""""
    S = 0
    n = len(centres)
    assert( n == len(L))
    for i in range(n) :
        for j in range(len(L[i])) :
            S += distance_e(centres[i], L[i][j][0]) ** 2
    return S ** (1/2)
            
# potentiellement pour montrer les limites de la norme euclidenne pourr differencier eles vecteurs 
def k_moyennes(L_images : ImagePathsEtiquetees, k, nb_tentatives = 10) ->  list[ImageVecteurs]:
    """
    L'obectif ici, c'est de trouver les tendances cachées au sein du set de données, 
    on va ainsi appliquer l'algorithme des k-moyennes sur 10 situations initiales aléatoirement choisies, 
    pour trouver la solution qui minimise la distance au sein d'un cluster.
    """
    #on stocke les resultats du k_means dans un dictionnaire 
    # on choisira a la fin celle qui minimise les distances au sein d'un chaques clusters
    tentatives = []
    L_vecteurs = traiter_les_images(L_images)
    for i in range(nb_tentatives): 
        point_rd = np.random.choice(len(L_vecteurs), k, replace = False)
        centres = traiter_les_images([L_images[i] for i in point_rd])
        centres = [centre[0] for centre in centres]
        #on continue l'iteration jusqu'a ce que la solution converge
        etapes = 0
        ancien_poids = 1
        poids = np.inf
        List = [[] for i in range(k)]
        while poids != 0 and np.abs( (ancien_poids - poids) / ancien_poids) > 0.05 and etapes < k * 10 : 
            etapes += 1
            # on attribue à chaque vecteur son cluster 
            for vect_etiq in L_vecteurs :
                L_distances = []
                for i in range(k) :
                    distance = distance_e(vect_etiq[0], centres[i])
                    L_distances.append((distance, i))
                # on attribue au cluster le plus proche
                (distance, cluster) = min(L_distances)
                List[cluster].append(vect_etiq)
            
            #on redefinit les centres comme les moyennes de chaques groupes
            
            for i in range(k) : 
                n = len(List[i])
                if n != 0 : 
                    centres[i] = np.zeros(centres[i].shape)
                    
                    for vect_etiq in List[i]: 
                        centres[i] += vect_etiq[0]
                    centres[i] /= n
                    
            ancien_poids = poids 
            poids = peser(List, centres)
            
        tentatives.append((poids, List[:]))
    recup = min(tentatives) 
    return recup[1][:]

    
def points_communs(L_images : ImagePathsEtiquetees, largeur = 50, bandes = 4, mini = 0.4) -> ImageVecteur : 
    """Renvoie les parties où se situent les caractéristiques clés de la liste d'images du même type d'images donnée en argument."""
    #on pre-traite les images 
    L_vecteurs = traiter_les_images(L_images)
    
    # quelques verifications d'usage
    assert(len(L_images) != 0)
    L = [L_vecteurs[0][0].shape == L_vecteurs[i][0].shape for i in range(len(L_images))]
    assert(False not in L)
    
    y = L_vecteurs[0][0].shape[0]
    renvoi = np.zeros(y)
    variabilite = np.zeros((y//largeur,))
    L_moyennes = np.zeros((y//largeur,))
    
    # on calcule les differences moyenne par rapport à la moyenne de tout les vecteurs... 
    # on le fait pour chaque sequence de "largeur" pixel    
    for i in range(y//largeur) :
        
        vect_moyen  = np.zeros(largeur)
        for i_vecteur in L_vecteurs :
            vect_moyen += i_vecteur[0][i * largeur : (i+1) * largeur]
        vect_moyen = vect_moyen / len(L_vecteurs)
        i_moyenne = np.mean(vect_moyen) 
        L_moyennes[i] = i_moyenne 
        
        # on calcule ainsi la distance cumuluée au vecteur moyen sur cette tranche de pixels 
        for i_vecteur in L_vecteurs :
            variabilite[i] += np.mean((i_vecteur[0][i* largeur : (i+1) * largeur] - vect_moyen)**2)
    maxi = max(L_moyennes)
    if maxi != 0 : 
        L_moyennes = L_moyennes / maxi
    # on recupere les endroits ou ça bouge le moins
    # si ils sont au moins 0.20, donc 20% du max, les vecteurs etant normalises
    nb_bandes = 0
    max_tours = (y //largeur) - 1
    # Sélectionner les 'bandes' segments les moins variables
    # avec une moyenne suffisante
    while nb_bandes < bandes and nb_bandes <= max_tours :
        i_min = np.argmin(variabilite)
        
        if L_moyennes[i_min] >= mini:
            renvoi[i_min * largeur: (i_min + 1) * largeur] = 1
            nb_bandes += 1 
        
        # On exclu ce segment pour les prochaines itérations
        variabilite[i_min] = np.inf
    return renvoi

def distance_e_partielle(vect_1 : ImageVecteur,vect_2 : ImageVecteur, parties : np.ndarray ) -> Distance : 
    """ on ne considère que les distances qui sont "importantes" pour les k moyennes"""
    assert(vect_1.shape == parties.shape)
    assert(vect_2.shape == parties.shape )
    return distance_e(vect_1*parties, vect_2 * parties)
  
    
def k_plus_proches_voisins_v2( L_images : ImagePathsEtiquetees , Chemin_Image : ImagePath, k) ->  list[Etiquete] :
    """
    cette fonction prend les chemins etiquetes, et le chemin de l'image, 
    elle traite les image, et definit leur distance ecludienne par rapport a l'image etudiee, 
    et renvoie l'etiquette la plus presente parmis ses k plus proches voisins    
    """
    L_Groupes = L_Groupes_defaut[:]
    vect_img = traiter_les_images([Chemin_Image])
    
    # on essaie de recuperer les groupes defini si il existe, sinon, on les fait avec k_moyennes
    if L_Groupes == [] : 
        L_Groupes = k_moyennes(L_images)
    else : 
        for G in L_Groupes : 
            G = traiter_les_images(G)
        
    for G in L_Groupes : 
        # on recupere d'abord les points commun du groupe G pour pouvoir utiliser la distance adaptee au groupe
        parties = points_communs(L_images)
        for vecteur_etiq in G :
            
            # on calcule les distances avec notre nouvelle fonction distance
            L_distances = []
            distance = distance_e_partielle(vecteur_etiq[0], vect_img, parties)
            
            
            # on rajoute ainsi la distance et l'etiquette dans la liste
            L_distances.append((distance, vecteur_etiq[1]))
    
    
    """ 
    A partir de la, on termine la fonction de la même façon que la v1, puisque des precautions ont ete prises : 
    entre autres, les "parties" ete calculee de façon a minimiser le plus la quantititee pour un certain groupe donne, 
    """
    
    L_distances.sort()
    
    # un dictionnaire pour stocker les occurences 
    dico_occurences = {} 
    
    for i in range(min(len(L_distances), k)) : 
        if L_distances[i][1] in dico_occurences : 
            dico_occurences[L_distances[i][1]] += 1 
        else : 
            dico_occurences[L_distances[i][1]] = 1
            
    # on renvoie ainsi la valeur la plus presentes parmis les k plus proches voisins 
    return max(dico_occurences, key=dico_occurences.get)


def voir_vecteur(chemin: str):
    """
    Affiche dans une bande horizontale de 50 lignes le milieu de l'image.
    """
    img = plt.imread(chemin)
    x, y, _ = img.shape
    x2 = x // 2
    bande = img[x2-25:x2+25, :, :] 
    gris = (
        0.299 * bande[:, :, 0]
      + 0.587 * bande[:, :, 1]
      + 0.114 * bande[:, :, 2]
    )  
    vecteur = gris.mean(axis=0)  # shape (y,)
    
    # Affichage en image agrandie comme une "ligne de pixels"
    plt.figure(figsize=(12, 2))
    plt.imshow(vecteur[np.newaxis, :], cmap='gray', aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.show()
    return vecteur 


def respresentation_2D(L_chemins_etiq : ImagePathsEtiquetees) -> ImageVecteur :
    couleurs = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    # on associe les couleurs avec les etiquettes 
    dico_assos = {}
    k_ieme_couleur = 0 
    
    for chemin_etiq in L_chemins_etiq :
        # pour chaque etiquette, une nouvelle couleur ! dans la mesure du possible.
        etiq = chemin_etiq[1]
        if etiq in dico_assos : 
            couleur = dico_assos[etiq] 
        else : 
            couleur = couleurs[k_ieme_couleur % len(couleurs)]
            k_ieme_couleur += 1
            dico_assos[etiq] = couleur 
        vecteur = traiter_les_images([chemin_etiq[0]])
        n = vecteur.shape
        x = distance_e(vecteur, np.zeros(n))
        y = distance_e(vecteur, np.ones(n))
        plt.scatter(x, y, color = couleur)

    plt.title("Spectres classés par cluster")
    plt.xlabel("Position")
    plt.ylabel("Intensité")
    plt.show()
    
    

