
# montrer les resultats : des fusions d'images avec les points communs avec
# dans la fonction test : diagramme d'incertitude

import numpy as np
import matplotlib.pyplot as plt


ImagePath = str
Etiquete = str 
Distance = float
ImageVecteur = np.ndarray


ImageEtiquetee = tuple[ImagePath, Etiquete]

ImagePaths = list[ImagePath]
ImagePaths_Etiquetees = list[ImageEtiquetee]
ImageVecteurs = list[ImageVecteur]


images = ["C:/Users/alban/Downloads/photo2.jpg", "C:/Users/alban/Downloads/phot32 (1).jpg"]

# un resultat de k_moyennes qui a été conservé
# on évite ainsi les aléas de l'algo des k-moyennes
L_Groupes_defaut = []





def convolution(vect: ImageVecteur, largeur = 50) -> ImageVecteur:
    """
    les images récupérées peuvent ne pas être parfaites à cause de decalages dans le dispositif,
    seulement, on ne paut pas laisser un décalage de quelques pixels venir rendre les mesures inexploitables.
    
    (on veut notamment stabiliser le clustering)
    """
    
    y = vect.shape[0]
    # paramètre à modifier si besoin 
    noyau = np.ones(y//largeur) / (y//largeur) 
    
    return np.convolve(vect, noyau, mode = 'same')
    



def voir_vecteur(chemin: ImagePath) -> ImageVecteur : 
    """
    cette fonction n'existe que pour faire un joli desssin sur 
    la presentation powerpoint
    """
    
    
    img = plt.imread(chemin)
    x, y, z = img.shape
    
    # on prend la ligne du milieu de l'image en la grisant 
    x2 = x // 2
    
    vect = np.zeros((50, y)) 
    for i in range(10) :
        for j in range(50) :
        # on recupere plusieurs lignes pour essayer de minimiser l'impact du bruit de la photo
            vect[j, :] += (  0.299 * img[x2 + i,:,0] 
                     + 0.587 * img[x2 + i,:,1] 
                     + 0.114 * img[x2 + i,:,2] )

    return convolution(vect)


def traiter_les_images( L_img: ImagePaths_Etiquetees ) -> ImageVecteurs :
    """ 
    cette fonction recupere les image à partir d'une liste contenant les chemins,  
    et elle transforme ces images en une liste de vecteurs sur lesquels on peut appliquer nos algorythmes
    """
    
    L_vecteurs = [] 
    
    for chemin_etiq in L_img :
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
        
        # on l'ajoute à la liste, avec la convolution
        L_vecteurs.append((convolution(vect[:]),chemin_etiq))
    
    return L_vecteurs

    

# cette fonctionn demontre l'ineficacité d'une simple norme euclidienne 
def points_communs(L_images : ImagePaths_Etiquetees, precision = 25, bandes = 4, mini = 0.2) -> ImageVecteur : 
    """
    on part avec une liste de listes d'images forunies par l'algo des k moyennes, 
    et on va essayer de leur trouver leurs points communs d'après les groupes determine par k-means, 
    on renvoie un vect contenant les parties quui sont particulères au spectres des groupes
    """
    # quelques verifications d'usage
    assert(len(L_images) != 0)
    
    L = [L_images[0].shape == L_images[i].shape for i in range(len(L_images))]
    assert(False not in L)
    
    # on transforme les images en vecteurs utilisable  
    L_vecteurs = traiter_les_images(L_images)
    
    y = L_vecteurs[0][0].shape 
    renvoi = np.zeros(y)
    variabilite = np.zeros(y//precision)
    L_moyennes = np.zeros(y//precision)
    
    # on calcule les differences moyenne par rapport à la moyenne de tout les vecteurs... 
    # on le fait pour chaque sequence de "precision" pixel    
    for i in range(y//precision) :
        
        vect_calcul = np.zeros(precision)
        for i_vecteur in L_vecteurs :
            vect_calcul += i_vecteur[i:i+precision]
            
        i_moyenne = np.mean(vect_calcul / len(L_vecteurs))
        L_moyennes[i] = i_moyenne
        vect_moyen = np.ones(precision) * i_moyenne / len(L_vecteurs)
        
        # on calcule ainsi la distance cumuluée au vecteur moyen sur cette tranche de pixels 
        for i_vecteur in L_vecteurs :
            variabilite[i] += np.mean((i_vecteur[i:i+precision] - vect_moyen)**2) ** (1/2)
        
    maxi_var = np.max(variabilite) 
    
    # on recupere les endroits ou ça bouge le moins
    # si ils sont au moins 0.20, donc 20%, les vecteurs etant normalises
    i = 0 
    while i < bandes : 
        arg_min = np.argmin(variabilite)
        if L_moyennes[arg_min] > mini :
            renvoi[arg_min : arg_min + 25] = np.ones(25)
            i += 1 
        variabilite[arg_min] = maxi_var + 1 
    
    return renvoi 

        
    
    
def distance(vect1 : ImageVecteur, vect2: ImageVecteur) -> Distance  : 
    """
    on calcule la distance euclidienne entre deux vecteurs,
    représentés par des tableaux à une dimension
    """
    y1 = vect1.shape[0]
    y2 = vect2.shape[0]
    if y1 != y2 : 
        raise ValueError("vecteurs impcompatibles")
    else : 
        S = 0 
        for yi in range(y1) :
            S += float( vect1[yi] - vect2[yi]) ** 2 
    return S ** (1/2)



def distance_c(vect_1 : ImageVecteur, vect_2 : ImageVecteur) -> Distance : 
    "on calcule la distance cosinus entre deux vecteurs représentés par des tableaux "
    y1 = vect_1.shape[0]
    y2 = vect_2.shape[0]
    if y1 != y2 : 
        raise ValueError("vecteurs impcompatibles")
    else : 
        S = 0 
        for yi in range(y1) :
            S += float( vect_1[yi] * vect_2[yi])  
            
    return 1 - S / ( distance_e(vect_1, np.zeros(y1)) * distance_e(vect_2, np.zeros(y2)))

    


def k_plus_proches_voisins_v1( L_images : ImagePaths_Etiquetees , Chemin_Image : ImagePath, k : int) ->  Etiquete :
    """
    cette fonction prend les chemins etiquetes, et le chemin de l'image, 
    elle traite les image, et definit leur distance ecludienne par rapport à l'image etudiee, 
    et renvoie l'etiquette la plus presente parmis ses k plus proches voisins    
    """
    vect_img = traiter_les_images([Chemin_Image])[0]
    L_distances = []
    
    for chemin_etiq in L_images :
        # on recupere le chemin dans chemin_etiqu, et on rajoute sa distance à la liste
        i_vecteur = traiter_les_images([chemin_etiq[0], "_"])[0]
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

def peser(L : list[ImagePaths_Etiquetees], centres, partie = None) -> float : 
    if partie == None : 
        partie = np.ones(L[0][0].shape)
    S = 0
    n = centres.shape[0]
    assert( n == len(L))
    for i in range(n) :
        for j in range(len(L[i])) :
            S += distance_e(centres[i], L[i][j]) ** 2
    return S ** (1/2)
            
        
    

# potentiellement pour montrer les limites de la norme euclidenne pourr differencier eles vecteurs 
def k_moyennes(L_images : ImagePaths_Etiquetees , Chemin_Image : ImagePath, k, nb_tentatives = 10) ->  list[ImageVecteurs]:
    """
    obectif ici, c'est de trouver les tendances cachées au sein du set de données, 
    on va ainsi appliquer l'algorithme des k-moyennes sur 10 situations initiales aléatoirement choisies, 
    pour trouver
    """
    #on stocke les resultats du k_means dans un dictionnaire 
    # on choisira a la fin celle qui minimise les distances au sein d'un chaques clusters
    tentatives = []
    L_vecteurs = traiter_les_images(L_images)
    
    for i in range(nb_tentatives): 
        
        centres = np.random.choice(len(L_vecteurs), k, replace = False)
        #on continue l'iteration jusqu'a ce que la solution converge
        etapes = 0
        ancien_poids = 0
        poids = np.inf
        while (poids != 0 or np.abs( (ancien_poids - poids) / poids) > 0.05) and etapes < k * 10 : 
            etapes += 1
            # on attribue à chaque vecteur son cluster 
            for vect_etiq in L_vecteurs :
                L = [[] for i in range(k)]
                L_distances = []
                for i in range(k) :
                    distance = distance_e(vect_etiq[0], centres[i])
                    L_distances.append((distance, vect_etiq[1]))
                # on attribue au cluster le plus proche
                (cluster, distance) = min(L_distances)
                L[cluster].append(vect_etiq)
    
            ancien_poids = poids 
            poids = peser(L, centres)
            
        tentatives.append((poids, L[:]))
    recup = min(tentatives) 
    return recup[1][:]       




def distance_e_partielle(vect_1 : ImageVecteur,vect_2 : ImageVecteur, parties : np.ndarray ) -> Distance : 
    """ on ne considère que les distances qui sont "importantes" pour les k moyennes"""
    assert(vect_1.shape == parties.shape)
    assert(vect_2.shape == parties.shape )
    return distance_e(vect_1*parties, vect_2 * parties)
    
    
    
def k_plus_proches_voisins_v2( L_images : ImagePaths_Etiquetees , Chemin_Image : ImagePath, k) ->  list[Etiquete] :
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




def tentative_respresentation_2D_v1(L_chemins_etiq : ImagePaths_Etiquetees) -> ImageVecteur :
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
        vecteur = traiter_les_images([chemin_etiq[0]])[0]
        n = vecteur.shape
        x = distance_e(vecteur, np.zeros(n))
        y = distance_e(vecteur, np.ones(n))
        plt.scatter(x, y, color = couleur)

    plt.title("Spectres classés par cluster")
    plt.xlabel("Position")
    plt.ylabel("Intensité")
    plt.show()
    
    

def tentative_respresentation_2D_v2(L_chemins_etiq : ImagePaths_Etiquetees) -> ImageVecteur :
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
        vecteur = traiter_les_images([chemin_etiq[0]])[0]
        n = vecteur.shape
        x = distance_e(vecteur, np.zeros(n))
        y = distance_e(vecteur, np.zeros(n))
        plt.scatter(x, y, color = couleur)

    plt.title("Spectres classés par cluster")
    plt.xlabel("Position")
    plt.ylabel("Intensité")
    plt.show()
    
