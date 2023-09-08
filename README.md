# ERAv1_Session16

This repository contains the PyTorch implementation of the Original Transformer Paper: [Transformer](https://arxiv.org/pdf/1706.03762.pdf), just with a few strategies that would increase the training process.

Refer to [S16.ipynb](/S16.ipynb) jupyter notebook for the training steps.

#### Strategies Used:
1. AMP [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
2. OneCyclePolicy to find the optimum learning rate and no. of epochs needed to reach it.
3. Dynamic padding for effectively reduce the training time.



### Training Logs
```
Total Parameters: 61847890
train_transformer(transformer_model, opus_fr, cfg, epochs=cfg['num_epochs'])
C:\Users\rohit\anaconda3\envs\tf\lib\site-packages\lightning_fabric\connector.py:555: UserWarning: 16 is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
  rank_zero_warn(
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name           | Type             | Params
----------------------------------------------------
0 | loss_criterion | CrossEntropyLoss | 0     
1 | model          | Transformer      | 61.8 M
----------------------------------------------------
61.8 M    Trainable params
0         Non-trainable params
61.8 M    Total params
247.392   Total estimated model params size (MB)
C:\Users\rohit\anaconda3\envs\tf\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:432: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
C:\Users\rohit\anaconda3\envs\tf\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:480: PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
  rank_zero_warn(
C:\Users\rohit\anaconda3\envs\tf\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:432: PossibleUserWarning: The dataloader, val_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Epoch 39: 100%
4975/4975 [08:42<00:00, 9.52it/s, v_num=1, train_loss=1.820]
*****************************************
    SOURCE: "Well, monseigneur?"
    TARGET: -- Bien, Monseigneur.
 PREDICTED: -- Eh bien , monsieur ?
*****************************************

*****************************************
    SOURCE: The boat made its way slowly towards the right shore.
    TARGET: Le bateau voguait lentement vers la rive droite.
 PREDICTED: Le vent se mit à coup , le Lièvre .
*****************************************

Epoch 0: train_loss=4.0544
*****************************************
    SOURCE: There was no one there now, but the Flemings and the rabble.
    TARGET: Il n’y avait plus que des flamands et de la canaille.
 PREDICTED: Il n ’ y avait pas maintenant , mais les femmes et les femmes .
*****************************************

*****************************************
    SOURCE: Finally he took his stand in a dark corner of the garden.
    TARGET: Enfin il alla se placer dans un coin obscur du jardin.
 PREDICTED: Enfin il prit sa tête noire du jardin .
*****************************************

Epoch 1: train_loss=3.7178
*****************************************
    SOURCE: What hatred she distills!
    TARGET: Que de haine elle distille!
 PREDICTED: Quelle haine !
*****************************************

*****************************************
    SOURCE: He took his meals in the kitchen alone, opposite the fire, on a little table brought to him all ready laid as on the stage.
    TARGET: Il prenait ses repas dans la cuisine, seul, en face du feu, sur une petite table qu’on lui apportait toute servie, comme au théâtre.
 PREDICTED: Il prit ses repas dans la cuisine , seul , en face du feu , sur une petite table qui lui donnait tout droit à la scène .
*****************************************

Epoch 2: train_loss=3.8802
*****************************************
    SOURCE: There could be no doubt that it had a very turn-up nose, much more like a snout than a real nose; also its eyes were getting extremely small for a baby: altogether Alice did not like the look of the thing at all.
    TARGET: Sans contredit son nez était très-retroussé, et ressemblait bien plutôt à un groin qu’à un vrai nez. Ses yeux aussi devenaient très-petits pour un bébé.
 PREDICTED: Il n ’ y avait aucun doute , ce n ’ était pas un nez , plus grand , comme un nez de vieux , mais aussi ses yeux étaient fort petits pour un petit enfant : tout en tout le regard de tout le monde .
*****************************************

*****************************************
    SOURCE: Fourrez-le donc, ce monsieur si habile, au fond d’une voiture de troisieme classe dans le chemin de fer souterrain et demandez-lui de vous énumérer les professions de ses compagnons de voyage ; je parierais mille contre un qu’il serait incapable de s’en tirer.
    TARGET: I should like to see him clapped down in a third class carriage on the Underground, and asked to give the trades of all his fellow-travellers. I would lay a thousand to one against him."
 PREDICTED: the , what sum , Signor , if , at one of the of , and ask you to him the of his of his , I could find a hundred yards upon which he would have no .
*****************************************

Epoch 3: train_loss=3.3073
*****************************************
    SOURCE: I began to feel the pangs of a violent hunger.
    TARGET: Je commençais à éprouver une faim violente.
 PREDICTED: Je commençai à trouver un violent faim .
*****************************************

*****************************************
    SOURCE: The knives were not sharpened, nor the floors waxed; there were iron gratings to the windows and strong bars across the fireplace; the little Homais, in spite of their spirit, could not stir without someone watching them; at the slightest cold their father stuffed them with pectorals; and until they were turned four they all, without pity, had to wear wadded head-protectors. This, it is true, was a fancy of Madame Homais'; her husband was inwardly afflicted at it.
    TARGET: Il y avait aux fenêtres des grilles en fer et aux chambranles de fortes barres. Les petits Homais, malgré leur indépendance, ne pouvaient remuer sans un surveillant derrière eux; au moindre rhume, leur père les bourrait de pectoraux, et jusqu’à plus de quatre ans ils portaient tous, impitoyablement, des bourrelets matelassés.
 PREDICTED: Les couteaux ne pas , ni les guides , il y avait des de fer aux fenêtres et de grosses barreaux ; le peu de peu , malgré leur esprit , ne pouvait les retenir , sans les revoir , ils les et ils se , jusqu ’ à quatre , sans qu ’ ils avaient fait , sans pitié , ils avaient eu la tête à Mme de , c ’ est une vraie tête de Mme Homais .
*****************************************

Epoch 4: train_loss=3.7455
*****************************************
    SOURCE: Elle ne pouvait s’accoutumer à ce malheur : son idole avait un défaut ; enfin dans un moment de bonne amitié elle demanda conseil au comte, ce fut pour celui-ci un instant délicieux et une belle récompense du mouvement honnête qui l’avait fait revenir à Parme.
    TARGET: She could not grow used to this disaster; her idol had a fault; finally, in a moment of frank friendship, she asked the Conte's advice; this was for him a delicious instant, and a fine reward for the honourable impulse which had made him return to Parma.
 PREDICTED: She could not fail to this request for the most ; at last moment , in a good moments of good friendship she asked the Conte ' s advice , that she was for a moment a moment of the great lady of the great lady of the great lady , who had left at Parma .
*****************************************

*****************************************
    SOURCE: "It is well for you that a low fever has forced you to abstain for the last three days: there would have been danger in yielding to the cravings of your appetite at first.
    TARGET: «Il est heureux, reprit-il, que la fièvre vous ait forcée à vous abstenir ces trois derniers jours: il y aurait eu du danger à céder dès le commencement à votre appétit vorace.
 PREDICTED: -- C ' est bien que la fièvre vous a forcée de vous traiter les trois jours ; il y aurait du danger pour céder aux de votre appétit .
*****************************************

Epoch 5: train_loss=3.6011
*****************************************
    SOURCE: Near three o'clock in the afternoon on July 6, fifteen miles south of shore, the Abraham Lincoln doubled that solitary islet at the tip of the South American continent, that stray rock Dutch seamen had named Cape Horn after their hometown of Hoorn.
    TARGET: Le 6 juillet, vers trois heures du soir, I'Abraham Lincoln, à quinze milles dans le sud, doubla cet îlot solitaire, ce roc perdu à l'extrémité du continent américain, auquel des marins hollandais imposèrent le nom de leur villa natale, le cap Horn.
 PREDICTED: Vers trois heures du soir , le 6 juillet , quinze kilometres au sud de l ' Abraham , le plus noir qui , au bout du continent , avait perdu le cap de l ' Amérique , après leur de .
*****************************************

*****************************************
    SOURCE: The hull of the "Speedy" was just beginning to issue from the water.
    TARGET: À ce moment, la coque du speedy commençait à se montrer au-dessus des eaux.
 PREDICTED: La coque du speedy commençait à s ' éloigner de l ' eau .
*****************************************

Epoch 6: train_loss=2.8747
*****************************************
    SOURCE: I had barely gotten up from my passably hard mattress when I felt my mind clear, my brain go on the alert.
    TARGET: A peine relevé de cette couche passablement dure, je sentis mon cerveau dégagé, mon esprit net.
 PREDICTED: Je m ' étais endormie à peine si j ' avais rencontré un matelas sur mes bras , quand je sentis mon cerveau s ' élancer .
*****************************************

*****************************************
    SOURCE: "Well, then," said Milady, "I confide in my brother; I will dare to--"
    TARGET: «Eh bien, dit Milady, je me fie à mon frère, j'oserai!»
 PREDICTED: -- Eh bien , dit Milady , je me fie à mon frère ; je osera ...
*****************************************

Epoch 7: train_loss=2.5106
*****************************************
    SOURCE: "No! but Mouquette promised to lend me half a franc."
    TARGET: —Non, c'est Mouquette qui a promis de me preter dix sous.
 PREDICTED: — Non ! mais la Mouquette m ' a promis de me prêter une moitié .
*****************************************

*****************************************
    SOURCE: — Non.
    TARGET: "No."
 PREDICTED: " No .
*****************************************

Epoch 8: train_loss=2.3297
*****************************************
    SOURCE: "Even money either way," cried the voice again.
    TARGET: -- Au pair sur les deux champions, répéta la voix.
 PREDICTED: -- De l ' argent , de nouveau , cria la voix .
*****************************************

*****************************************
    SOURCE: "Perfectly."
    TARGET: -- Parfaitement.
 PREDICTED: -- Parfaitement .
*****************************************

Epoch 9: train_loss=2.7592
*****************************************
    SOURCE: "You ought to know."
    TARGET: -- Vous devez bien le savoir.
 PREDICTED: -- Vous devez le savoir .
*****************************************

*****************************************
    SOURCE: All the same, I callit good luck, jolly good luck!"
    TARGET: C'est égal, en voilàune veine, une rude veine!
 PREDICTED: Je ne regrette rien , ma chérie ! »
*****************************************

Epoch 10: train_loss=2.0522
*****************************************
    SOURCE: "My gas-burner, which I forgot to turn off, and which is at this moment burning at my expense.
    TARGET: -- Mon bec de gaz que j'ai oublié d'éteindre et qui brûle à mon compte.
 PREDICTED: « Mon bec de gaz , que j ' ai oublié de me retourner , et ce qui est en ce moment dévoré .
*****************************************

*****************************************
    SOURCE: Day dawned; Pencroft and his companions immediately proceeded to survey the dwelling.
    TARGET: Pencroff et ses compagnons procédèrent immédiatement à l'examen de l'habitation.
 PREDICTED: Le jour parut , Pencroff et ses compagnons se aussitôt à l ' heure de la demeure .
*****************************************

Epoch 11: train_loss=1.8327
*****************************************
    SOURCE: Faster and faster yet they raced, the hoofs rattling like castanets, the yellow manes flying, the wheels buzzing, and every joint and rivet creaking and groaning, while the curricle swung and swayed until I found myself clutching to the side-rail.
    TARGET: Les crinières jaunes voltigeaient, les roues bourdonnaient. Toutes les jointures, tous les rivets craquaient, gémissaient pendant que la voiture oscillait et se balançait au point que je dus me cramponner à la barre de côté.
 PREDICTED: Plus ils couraient , les pattes , des robinets de poules , des de buissons jaunes , les roues , , and chaque , en se , et en fermant la route , en me vers le côté de la barrière .
*****************************************

*****************************************
    SOURCE: "Listen, then, Jane Eyre, to your sentence: to-morrow, place the glass before you, and draw in chalk your own picture, faithfully, without softening one defect; omit no harsh line, smooth away no displeasing irregularity; write under it, 'Portrait of a Governess, disconnected, poor, and plain.'
    TARGET: «Jane Eyre, écoute donc ta sentence: demain tu prendras une glace et tu feras fidèlement ton portrait, sans omettre un seul défaut, sans adoucir une seule ligne trop dure, sans effacer une seule irrégularité déplaisante; tu écriras en dessous: «Portrait d'une gouvernante laide, pauvre et sans famille.»
 PREDICTED: -- Écoutez - vous donc , Jane Eyre , pour vous parler ; demain , au verre de vous et vous dans votre image par un défaut dur , sans vous donner un défaut ; non pas de , n ' en point , d ' écrire sous elle , elle est , pauvre et , se .
*****************************************

Epoch 12: train_loss=2.0704
*****************************************
    SOURCE: "You saw your wi--"
    TARGET: «Tu as vu ta fem...
 PREDICTED: -- Vous avez vu l ' argent avec vous ...
*****************************************

*****************************************
    SOURCE: The attack of gout was prolonged by the wintry weather and lasted for some months.
    TARGET: L’attaque de goutte fut prolongée par les grands froids de l’hiver et dura plusieurs mois.
 PREDICTED: L ’ attaque de goutte coulait rapidement par l ’ immense saison , et il tomba pendant quelques mois .
*****************************************

Epoch 13: train_loss=2.0462
*****************************************
    SOURCE: A favourite subject of conversation was afforded by the experiences of old Michaud who was plied with questions respecting the strange and sinister adventures with which he must have been connected in the discharge of his former functions.
    TARGET: Un des grands sujets de conversation était de parler au vieux Michaud de ses anciennes fonctions, de le questionner sur les étranges et sinistres aventures auxquelles il avait dû être mêlé.
 PREDICTED: On causait une conversation avec le favori de la vieille : il faisait des questions , des aventures étranges dont il avait dû occuper la partie des anciennes .
*****************************************

*****************************************
    SOURCE: Par une bizarrerie de malade, dès que le général Fabio Conti avait pu parler, il avait fait monter deux cents soldats dans cet ancien corps de garde abandonné depuis un siècle.
    TARGET: By a sick man's whim, as soon as General Fabio Conti was able to speak, he had ordered up two hundred soldiers into this old guard-room, disused for over a century.
 PREDICTED: By a of ill , at which General Fabio Conti had been speaking , he had put on two hundred soldiers in this old guard - room of an old guard in the old guard - room .
*****************************************

Epoch 14: train_loss=1.8917
*****************************************
    SOURCE: From sentence to sentence they came to mutual reproaches about this drowning business at Saint-Ouen, casting the crime in the face of one another.
    TARGET: De parole en parole, ils en arrivaient à se reprocher la noyade de Saint-Ouen; alors ils voyaient rouge, ils s'exaltaient jusqu'à la rage.
 PREDICTED: De sentence de sentence , on prit des reproches à parler , de ce train de à Saint - Ouen au visage de l ' un à l ' autre .
*****************************************

*****************************************
    SOURCE: Then he was silent.
    TARGET: Alors il se tut.
 PREDICTED: Alors il se tut .
*****************************************

Epoch 15: train_loss=1.9467
*****************************************
    SOURCE: It was all that remained of the structure of Granite House!
    TARGET: C'était tout ce qui restait du massif de Granite-House!
 PREDICTED: C ' était tout ce qui restait du monument de Granite - House !
*****************************************

*****************************************
    SOURCE: Think of the ignominy with which he will drive you from the house; all Verrieres, all Besancon will ring with the scandal.
    TARGET: Songe que c’est avec ignominie qu’il te chassera de sa maison ; tout Verrières, tout Besançon parleront de ce scandale.
 PREDICTED: Songez à l ’ ignominie qu ’ il vous chassera ; tout Verrières , tout Verrières , va sonner Besançon avec le scandale .
*****************************************

Epoch 16: train_loss=1.9910
*****************************************
    SOURCE: Enfin le domestique vint lui annoncer que Mlle Clélia Conti était disposée à le recevoir.
    TARGET: At length the servant came to inform him that Signorina Clelia Conti was willing to receive him.
 PREDICTED: At length the servant came to him that Mlle Clelia Conti was beginning to fall .
*****************************************

*****************************************
    SOURCE: She allowed the vessel to pass Lorient and Brest without repeating her request to the captain, who, on his part, took care not to remind her of it.
    TARGET: Elle laissa donc passer Lorient et Brest sans insister près du capitaine, qui, de son côté, se garda bien de lui donner l'éveil.
 PREDICTED: Elle laissait aller le navire au , sans qu ' elle ni lui ni demander la demander au capitaine , qui , de sa part , n ' en voulait pas la rappeler .
*****************************************

Epoch 17: train_loss=1.9044
*****************************************
    SOURCE: "All."
    TARGET: --Toutes.
 PREDICTED: -- Tout .
*****************************************

*****************************************
    SOURCE: Elizabeth's eyes were fixed on her with most painful sensations, and she watched her progress through the several stanzas with an impatience which was very ill rewarded at their close; for Mary, on receiving, amongst the thanks of the table, the hint of a hope that she might be prevailed on to favour them again, after the pause of half a minute began another.
    TARGET: Elizabeth l’écouta chanter plusieurs strophes avec une impatience qui ne s’apaisa point a la fin du morceau ; car quelqu’un ayant exprimé vaguement l’espoir de l’entendre encore, Mary se remit au piano.
 PREDICTED: Les yeux d ’ Elizabeth étaient fixés sur elle , et , parmi ses doigts , elle ne voyait pas , par cette lueur de , de ces cris d ’ impatience qui les , c ’ était elle , a la vue de Marie , la table de la table , – l ’ idée qu ’ elle ferait de l ’ espoir qu ’ elle pourrait leur donner une demi - minute , apres une demi - minute .
*****************************************

Epoch 18: train_loss=1.8376
*****************************************
    SOURCE: "Well," said Pencroft, "this bay would make admirable roads, in which a whole fleet could lie at their ease!"
    TARGET: «Voilà, dit Pencroff, un bout de mer qui ferait une rade admirable, où des flottes pourraient évoluer à leur aise!
 PREDICTED: « Eh bien , dit Pencroff , cette baie est une admirable rade , et dont toute une flotte à l ' aise !
*****************************************

*****************************************
    SOURCE: "Monsieur de Treville?" said the stranger, becoming attentive, "he put his hand upon his pocket while pronouncing the name of Monsieur de Treville?
    TARGET: -- M. de Tréville? dit l'inconnu en devenant attentif; il frappait sur sa poche en prononçant le nom de M. de Tréville?...
 PREDICTED: -- M . de Tréville ? dit l ' inconnu en s ' animant , il a mis sa main sur sa poche , en poussant le nom de M . de Tréville ?
*****************************************

Epoch 19: train_loss=1.8544
*****************************************
    SOURCE: I promise not to go without you.'
    TARGET: Je te promets de ne pas repartir sans toi.
 PREDICTED: Je ne vous demande pas non .
*****************************************

*****************************************
    SOURCE: "Shall I bring him alone?"
    TARGET: -- L'amènerai-je seul?
 PREDICTED: -- - je l ' amener seul ?
*****************************************

Epoch 20: train_loss=1.8999
*****************************************
    SOURCE: I made some attempts to draw her into conversation, but she seemed a person of few words: a monosyllabic reply usually cut short every effort of that sort.
    TARGET: Je tâchai plusieurs fois d'entrer en conversation avec elle; mais elle n'était pas causante. Généralement une réponse monosyllabique coupait court à tout entretien.
 PREDICTED: Je fis quelques consolations à la conversation . Elle me parut avoir une réponse ordinaire , et un mouvement de fatigue coupée chaque instant de ce genre .
*****************************************

*****************************************
    SOURCE: "Well?"
    TARGET: -- Eh bien?
 PREDICTED: -- Eh bien ?
*****************************************

Epoch 21: train_loss=1.9365
*****************************************
    SOURCE: It was the coffer which Ayrton had saved at the risk of his life, at the very instant that the island had been engulfed, and which he now faithfully handed to the engineer.
    TARGET: C'était le coffret qu'Ayrton avait sauvé au péril de sa vie, au moment où l'île s'engloutissait, et qu'il venait fidèlement remettre à l'ingénieur.
 PREDICTED: C ' était le coffret que fit au risque de sa vie , au moindre instant que l ' île avait été et que celui qu ' il avait présent avait abandonné à l ' ingénieur .
*****************************************

*****************************************
    SOURCE: "You may speak," said Madame Hennebeau complacently.
    TARGET: —Vous pouvez parler, dit madame Hennebeau complaisamment.
 PREDICTED: — Vous pouvez parler , dit madame Hennebeau avec complaisance .
*****************************************

Epoch 22: train_loss=1.9764
*****************************************
    SOURCE: It was impossible to tell what rocks we were passing: the tunnel, instead of tending lower, approached more and more nearly to a horizontal direction, I even fancied a slight rise.
    TARGET: Le tunnel, au lieu de s'enfoncer dans les entrailles du globe, tendait à devenir absolument horizontal. Je crus remarquer même qu'il remontait vers la surface de la terre.
 PREDICTED: Il n ' était pas possible de dire ce rocher : l ' tunnel , sans hâte , s ' approcher de plus en plus , s ' approcher et ne presque plus à une direction presque fin .
*****************************************

*****************************************
    SOURCE: It was one of those spectres, half light, half shadow, such as one beholds in dreams and in the extraordinary work of Goya, pale, motionless, sinister, crouching over a tomb, or leaning against the grating of a prison cell.
    TARGET: C’était un de ces spectres mi-partis d’ombre et de lumière, comme on en voit dans les rêves et dans l’œuvre extraordinaire de Goya, pâles, immobiles, sinistres, accroupis sur une tombe ou adossés à la grille d’un cachot.
 PREDICTED: C ’ était une de ces spectres , à demi - obscurité , comme on voit dans les songes extraordinaire , et on voit dans l ’ opération d ’ une pâle et immobile , accroupie devant une tombe , ou se penchant sur une grille à la grille .
*****************************************

Epoch 23: train_loss=1.8470
*****************************************
    SOURCE: What do his faults, his absurdities matter?
    TARGET: Qu’importent ses défauts, ses ridicules ?
 PREDICTED: Que devint - il ses défauts .
*****************************************

*****************************************
    SOURCE: But neither there, nor in any other part of Mount Franklin, did the colonists find any traces of him of whom they were in search.
    TARGET: Mais ni là, ni en aucune autre partie du mont Franklin, les colons ne trouvèrent les traces de celui qu'ils cherchaient.
 PREDICTED: Mais ni lui , ni aucun autre part du mont Franklin , n ' en firent aucune trace de lui que l ' on cherchait .
*****************************************

Epoch 24: train_loss=1.8189
*****************************************
    SOURCE: It was evident that the bellringer was to serve the archdeacon for a given time, at the end of which he would carry away the latter's soul, by way of payment.
    TARGET: Il était évident que le sonneur devait servir l’archidiacre pendant un temps donné au bout duquel il emporterait son âme en guise de paiement.
 PREDICTED: Il était évident que le sonneur de cloches était de servir l ’ archidiacre à temps , au bout de lequel il l ’ âme de celui - ci , par manière de gagner .
*****************************************

*****************************************
    SOURCE: M. de Caylus asserted that he had been credited with the determination to propose for the hand of Mademoiselle de La Mole (to whom the Marquis de Croisenois, who was heir to a Dukedom with an income of one hundred thousand livres, was paying court).
    TARGET: M. de Caylus prétendait qu’on lui avait donné la volonté de demander en mariage Mlle de La Mole (à laquelle le marquis de Croisenois, qui devait être duc avec cent mille livres de rente, faisait la cour).
 PREDICTED: M . de Caylus qu ’ il lui avait appris à faire cette résolution pour lui de proposition pour la main de Mlle de La Mole ( à qui le marquis de Croisenois , qui était héritier à un commerce de cent mille livres de rente , on peut faire la cour ).
*****************************************

Epoch 25: train_loss=1.9091
*****************************************
    SOURCE: "Bread! bread! bread!"
    TARGET: —Du pain! du pain! du pain!
 PREDICTED: — Du pain ! du pain ! du pain !
*****************************************

*****************************************
    SOURCE: Since 1815 he has blushed at his connection with industry: 1815 made him Mayor of Verrieres.
    TARGET: Depuis 1815 il rougit d’être industriel : 1815 l’a fait maire de Verrières.
 PREDICTED: Depuis qu ’ il a rougit de son rapport avec l ’ assurance de l ’ industrie et qui le fait voir M . le maire de Verrières .
*****************************************

Epoch 26: train_loss=1.7667
*****************************************
    SOURCE: The latter--a pretty girl of about twenty or twenty-two years, active and lively, the true SOUBRETTE of a great lady--jumped from the step upon which, according to the custom of the time, she was seated, and took her way toward the terrace upon which d’Artagnan had perceived Lubin.
    TARGET: Cette dernière, jolie fille de vingt à vingt-deux ans, alerte et vive, véritable soubrette de grande dame, sauta en bas du marchepied, sur lequel elle était assise selon l'usage du temps, et se dirigea vers la terrasse où d'Artagnan avait aperçu Lubin.
 PREDICTED: Celle - ci , une jolie fille de vingt ou vingt - deux ans , active et fine , la vraie soubrette d ' une grande dame , sauta de la marche sur laquelle , selon la coutume de l ' époque , elle fut assise , et se dirigea vers la terrasse que d ' Artagnan avait aperçu .
*****************************************

*****************************************
    SOURCE: Our family, the Stones, have for many generations belonged to the navy, and it has been a custom among us for the eldest son to take the name of his father's favourite commander.
    TARGET: Notre famille, les Stone, était depuis bien des générations vouée à la marine et il était de tradition, chez nous, que l'aîné portât le nom du commandant favori de son père.
 PREDICTED: Notre famille , les Stone , ont pour plusieurs générations d ' générations de sa marine , et il a été d ' une habitude parmi nous pour le fils aîné de son père favori .
*****************************************

Epoch 27: train_loss=1.7472
*****************************************
    SOURCE: Have you yesterday's Times, Watson?"
    TARGET: Avez-vous le Times d’hier, Watson ?
 PREDICTED: Vous avez le Times , Watson ?
*****************************************

*****************************************
    SOURCE: She told him one day that she was reading d'Aubigne's _History_, and Brantome.
    TARGET: Elle lui dit un jour qu’elle lisait l’histoire de d’Aubigné et Brantôme.
 PREDICTED: Elle lui dit un jour qu ’ elle lisait en , .
*****************************************

Epoch 28: train_loss=1.8868
*****************************************
    SOURCE: I trust that he is kind to her.
    TARGET: J’espère qu’il est gentil avec elle.
 PREDICTED: J ’ espère qu ’ il lui est bon .
*****************************************

*****************************************
    SOURCE: "Her lover."
    TARGET: -- Son amant.
 PREDICTED: -- Son amant .
*****************************************

Epoch 29: train_loss=1.7362
*****************************************
    SOURCE: I have been spitting blood all the time.
    TARGET: "Je ne cesse de cracher le sang.
 PREDICTED: J ' ai pleuré le sang à la fois .
*****************************************

*****************************************
    SOURCE: At times, he imagined a streak of blood was running down his chest, and would bespatter his white waistcoat with crimson.
    TARGET: Il s'imaginait par moments qu'un filet de sang lui coulait sur la poitrine et allait tacher de rouge la blancheur de son gilet.
 PREDICTED: Par moments , il imaginait le pas , un sang - froid lui sortait par la poitrine , et allait son gilet blanc .
*****************************************

Epoch 30: train_loss=1.7893
*****************************************
    SOURCE: "True.
    TARGET: -- Juste.
 PREDICTED: -- C ' est vrai .
*****************************************

*****************************************
    SOURCE: Later on I came to know in minute detail what had happened out there. . .
    TARGET: Plus tard, j’ai su par le menu détail tout ce qui s’était passé là-bas…
 PREDICTED: Plus tard , je revins à moi dans les moments où s ' était passé la , ce qui était arrivé .
*****************************************

Epoch 31: train_loss=1.8169
*****************************************
    SOURCE: 'Tis a treasure.
    TARGET: C’est un trésor.
 PREDICTED: C ’ est un trésor .
*****************************************

*****************************************
    SOURCE: But the other was too tricky for him, and was on him like a shot.
    TARGET: Mais l'autre était un vieux routier, et il fondit sur lui comme un boulet.
 PREDICTED: Mais l ’ autre était trop fort pour lui , et il était comme un coup .
*****************************************

Epoch 32: train_loss=1.7450
*****************************************
    SOURCE: "Hold, hold, hold!" said Athos, wit his quiet tone; "that throw of the dice is extraordinary. I have not seen such a one four times in my life. Two aces!"
    TARGET: «Tiens, tiens, tiens, dit Athos avec sa voix tranquille, ce coup de dés est extraordinaire, et je ne l'ai vu que quatre fois dans ma vie: deux as!»
 PREDICTED: « Arrêtez , monsieur ! dit Athos , faites un ton tranquille , celui des dés sont extraordinaire ; je n ' ai pas vu un si bien quatre fois dans ma vie !
*****************************************

*****************************************
    SOURCE: So I discoursed that point with my governess, and she went and waited upon the captain, and told him that she hoped ways might be found out for her two unfortunate cousins, as she called us, to obtain our freedom when we came into the country, and so entered into a discourse with him about the means and terms also, of which I shall say more in its place; and after thus sounding the captain, she let him know, though we were unhappy in the circumstances that occasioned our going, yet that we were not unfurnished to set ourselves to work in the country, and we resolved to settle and live there as planters, if we might be put in a way how to do it.
    TARGET: Je parlai à ce sujet avec ma gouvernante, et elle alla trouver le capitaine, à qui elle dit qu'elle espérait qu'on pourrait trouver moyen d'obtenir la liberté de ses deux malheureux cousins, comme elle nous appelait, quand nous serions arrivés par delà la mer; puis s'enquit de lui quelles choses il était nécessaire d'emporter avec nous, et lui, en homme d'expérience, lui répondit:
 PREDICTED: Je donc que ce point avec ma gouvernante et elle l ' attendait sur le capitaine et lui dis qu ' elle pouvait être en attendant ses deux pauvres cousines , comme elle nous appelait pour nous aider lorsque nous entra dans la campagne , et nous lui dis avec le fait dont je saurai l ' espèce et les moyens de lui dire ; et après quoi je le capitaine , ainsi que nous puissions le saurions en avoir fait remarquer , et que nous ne saurions où nous notre vaisseau en route .
*****************************************

Epoch 33: train_loss=1.7092
*****************************************
    SOURCE: She thanked Suzanne for her attention. Although weakened, she talked, and had ceased wandering, but she spoke in a voice so full of sadness that at moments she was half choked.
    TARGET: Elle remercia Suzanne de ses soins, elle parla, affaiblie, ne délirant plus, pleine d'une tristesse qui l'étouffait par moments.
 PREDICTED: Elle sa attention ; elle s ' ennuyait , elle parlait , elle avait cessé , mais elle parlait d ' une voix si pleine de tristesse que les instants elle étouffait .
*****************************************

*****************************************
    SOURCE: The next day, as Candide was walking out, he met a beggar all covered with scabs, his eyes sunk in his head, the end of his nose eaten off, his mouth drawn on one side, his teeth as black as a cloak, snuffling and coughing most violently, and every time he attempted to spit out dropped a tooth.
    TARGET: Le lendemain, en se promenant, il rencontra un gueux tout couvert de pustules, les yeux morts, le bout du nez rongé, la bouche de travers, les dents noires, et parlant de la gorge, tourmenté d'une toux violente, et crachant une dent à chaque effort.
 PREDICTED: Le lendemain Candide , au bout de ses dents , trouva une mendiante , tombée la tête à la tête , le bout du nez , l ' échine lui servait à côté , les dents noires comme un manteau , et qui fasse un manteau léger avec violence , et il essayait à cracher dehors .
*****************************************

Epoch 34: train_loss=1.6591
*****************************************
    SOURCE: In a few minutes the water reached 100 degrees centigrade.
    TARGET: En quelques minutes, cette eau avait atteint cent degrés.
 PREDICTED: En quelques minutes , l ' eau arrivait à cent degrés .
*****************************************

*****************************************
    SOURCE: "One has to; but he wants more than that."
    TARGET: --Il le faut bien, mais il veut encore autre chose.
 PREDICTED: -- On le veut , mais il en veut davantage .
*****************************************

Epoch 35: train_loss=1.7919
*****************************************
    SOURCE: My father was sitting with staring eyes, and his forgotten pipe reeking in his hand.
    TARGET: Mon père restait immobile, les yeux fixes, oubliant la pipe fumante qu'il tenait à la main.
 PREDICTED: Mon père , assis les yeux fixes , trouva sa pipe dans la main .
*****************************************

*****************************************
    SOURCE: I will sell the rest of what I do not want, and with this alone I will make two thousand francs a year.
    TARGET: Je vendrai le superflue de ce que j'ai, et avec cette vente seule, je me ferais deux mille livres par an.
 PREDICTED: Je vendrai encore aux autres où je ne veux pas , et avec ces seuls , je ferai mille francs par an .
*****************************************

Epoch 36: train_loss=1.7323
*****************************************
    SOURCE: "I cannot proceed without some investigation into what has been asserted, and evidence of its truth or falsehood."
    TARGET: «Je ne puis pas continuer avant d'avoir examiné ce qui vient d'être dit.
 PREDICTED: – Je ne puis aller sans nécessité de comprendre quoi faire qui a apparence , et de sa réalité ou de mensonge .
*****************************************

*****************************************
    SOURCE: Hans pointed with his finger at a dark mass six hundred yards away, rising and falling alternately with heavy plunges.
    TARGET: Hans montre du doigt, à une distance de deux cents toises, une masse noirâtre qui s'élève et s'abaisse tour à tour.
 PREDICTED: Hans montrait au doigt une masse sombre de six cents yards au pas , qui se releva et se de laquais .
*****************************************

Epoch 37: train_loss=1.7227
*****************************************
    SOURCE: Mme. Roland had not emptied her glass and was gazing at her son Jeanwith sparkling eyes; happiness had brought a colour to her cheeks.
    TARGET: Mme Roland n'avait point vidé son premier verre, et, rose de bonheur, leregard brillant, elle contemplait son fils Jean.
 PREDICTED: Mme Roland n ' avait pas vidé son verre et regardait son fils ses yeux vifs , car le bonheur lui avait fait des joues .
*****************************************

*****************************************
    SOURCE: "In the Mediterranean!" I exclaimed.
    TARGET: -- Dans la Méditerranée ! m'écriai-je.
 PREDICTED: -- Dans la Méditerranée ? m ' écriai - je .
*****************************************

Epoch 38: train_loss=1.7922
*****************************************
    SOURCE: His faith in the engineer was complete; nothing could disturb it.
    TARGET: Sa foi dans l'ingénieur était absolue. Rien n'eût pu la troubler.
 PREDICTED: Sa foi , en l ' ingénieur était complète , rien ne la .
*****************************************

*****************************************
    SOURCE: He had rested his elbow upon the open volume of Honorius d'Autun, ~De predestinatione et libero arbitrio~, and he was turning over, in deep meditation, the leaves of a printed folio which he had just brought, the sole product of the press which his cell contained.
    TARGET: Il avait appuyé son coude sur le livre tout grand ouvert d’Honorius d’Autun, De prædestinatione et libero arbitrio, et il feuilletait avec une réflexion profonde un in-folio imprimé qu’il venait d’apporter, le seul produit de la presse que renfermât sa cellule.
 PREDICTED: Il avait son coude sur la rive ouverte de d ’ Autun , à et à ; il fut par - dessus , dans une profonde méditation , il venait de faire apporter un tableau au - dessous de la porté , le produit de la presse .
*****************************************

Epoch 39: train_loss=1.7974

Testing DataLoader 0: 100%
8844/8844 [00:11<00:00, 786.26it/s]
*****************************************
    SOURCE: "Yes, I see," Ned replied, growing more interested. "Because the water surrounds me but doesn't penetrate me."
    TARGET: -- Oui, je comprends, répondit Ned, devenu plus attentif, parce que l'eau m'entoure et ne me pénètre pas.
 PREDICTED: -- Oui , je le vois , répondit Ned , mais l ' eau qui m ' entoure , mais qui ne me tourmente pas .
*****************************************

*****************************************
    SOURCE: She held out her tin to him.
    TARGET: Et elle lui tendit sa gourde.
 PREDICTED: Elle lui tendit sa gourde .
*****************************************
```
#### Sample Predictions
    SOURCE: "In the Mediterranean!" I exclaimed.
    TARGET: -- Dans la Méditerranée ! m'écriai-je.
 PREDICTED: -- Dans la Méditerranée ? m ' écriai - je .

#### One Cycle Policy
OCP (One Cycle Policy) was used to get the optimum learning_rate.
```
scheduler = OneCycleLR(
    optimizer,
    max_lr=1E-04,
    steps_per_epoch=len(train_data_loader),
    epochs=40,
    pct_start=0.125,
    div_factor=10,
    three_phase=True,
    final_div_factor=10,
    anneal_strategy='linear'
)
```



 #### Training Loss:
 The training loss function over the 40 epochs is as follows. As you can see on the tensorboard that the final loss achieved is 1.79
 ![training_loss_final](/images/training_loss_final.png)

 ![training_loss](/images/training_loss.png)
