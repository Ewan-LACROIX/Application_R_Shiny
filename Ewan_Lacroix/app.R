
library(shiny)
library(tidyverse)
library(corrplot)
library(knitr)
library(texreg)
library(margins)
library(pROC)
library(haven)
library(shinythemes)
library(DT)
library(broom)
library(bslib)

ui <- shinyUI(
  navbarPage(theme = shinytheme("flatly"),
    title = "Credit Scoring",
    
    tabPanel(title = "Présentation",
             div(style = "text-align: center;",img(src = "esa.png",height = "200px", width = "800px")),
             hr(),
             
                 fluidRow(
                   div(style = "text-align: center;",
                    column(6,
                           h3("Préambule"),
                           p("Cette application est réalisée dans le cadre de l'évaluation de la matière",
                             a(href = "https://www.master-esa.fr/wp-content/uploads/2024/05/NDOYE-Nouvelles-technologies-sous-R_MaJ-2022.pdf",strong("R Avancé")),
                             "au sein du ", a(href = "https://www.master-esa.fr/",strong("Master ESA")), 
                             "et a été réaliséé en 2025. Les analyses et interprétations futures sont à prendre 
                             avec des pincettes puisqu'elles ne sont en soit l'intérêt de l'évaluation."),
                           h3("Objectifs"),
                           p("Le but de cette application est de pouvoir effectuer du Credit Scoring"),
                           br(),
                           p("Le",strong("credit scoring"), "est une méthode utilisée pour évaluer la solvabilité 
                             d'un individu ou d'une entreprise en fonction de diverses données financières 
                             et personnelles. L'objectif est de prédire la probabilité qu'un emprunteur rembourse 
                             son crédit en temps et en heure. Le score attribué sert à déterminer le risque 
                             associé à un prêt et à prendre des décisions sur l'octroi de crédits."),
                           br(),
                           p("Pour ce faire, nous utilisons une base de données contenant des informations
                             sur le defaut de paiement de clients provenant de Taiwan d'Avril à Septembre 2005.
                             Cette base provient du site kaggle et peut être récupérée en cliquant sur ce"
                             ,a(href = "https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?s",strong("lien"))
                             ,'.')
                          )
                      ),
                    column(6,
                           div(style = "text-align: center;",
                             h3("Parcours"),
                             p("Cette application vous permet de voir en revue toutes les étapes necessaires aux
                               réalisations des objectifs. Le plan est le suivant :"),
                           ),
                           tags$ul(
                             tags$li("Rappels théoriques"),
                             tags$li("Visualisation de nos données"),
                             tags$li("Modélisation, evaluation"),
                             tags$li("Prédictions"),
                             tags$ul(
                              tags$li("Renseignements"),
                              tags$li("Résultats")
                              )
                            ),
                           div(style = "text-align: center;",img(src = "univ.png",height = "200px", width = "400px"))
                           )
                   
                ),
             div(style = "position: fixed; bottom: 0; width: 100%; text-align: center; 
             background-color: #f8f9fa; padding: 10px; color: grey;",
                 "Réalisé par Lacroix Ewan")
              
            ),
    
    navbarMenu("Théorie",
    tabPanel(title = "Modélisation",
             fluidRow(
               div(style = "text-align: center;",
                   p("Les rappels suivants sont issus du cours de Mr. Hurlin disponible", a(href = "https://sites.google.com/view/christophe-hurlin/teaching-resources/variables-qualitatives?authuser=0)","ici"),".")
                  ),
               column(4,
             h3("Spécification du modèle"),
             p("Notre variable d'interêt ne peut prendre que 2 valeurs, elle est donc
              dichotomique. On notera Yi la variable à expliquer pour l'individu
              i . L'évènement sera defaut."),
             withMathJax("$$ y_i = \\begin{cases} 
                         1 & \\text{si l'individu $i$ connaît l'évènement} \\\\
                         0 & \\text{sinon}
                         \\end{cases} $$"),
             p("Avec notre modèle, nous ne pourrons directement définir si un individu
              connait l'evenement, nous modéliserons la probabilité que celui-ci
              connaisse l'évènement."),
             withMathJax("$$\\pi_i=Prob(y_i=1|x_i)=F(x_i\\beta)$$"),
             p("La fonction F(.) représente fonction de répartition (lien) qui diffère
              selon le modèle dichotomique choisi. On a Xi le vecteur des variables
              explicatives et Beta le vecteur des coefficients associés."),
             withMathJax("$$F(x_i\\beta)=\\frac{e^{x_i\\beta}}{1+e^{x_i\\beta}}=\\frac{1}{1+e^{-x_i\\beta}}=\\Lambda(x_i\\beta)$$"),
             p("Nous pourrons donc bien avoir en sortie des probabilités et nous pouvons
                re-spécifier via une variable latente y*:"),
             withMathJax("$$y_i=\\begin{cases} 1 &\\text{ si $y_i^{\\star}=x_i\\beta+\\varepsilon_i > \\gamma$}\\\\0 &\\text{sinon}\\end{cases}$$"),
             withMathJax("$$y_i=\\begin{cases} 1 &\\text{avec une probabilité $\\pi_i$}\\\\0 &\\text{avec une probabilité $1-\\pi_i$}\\end{cases}$$"),
               ),
             column(4,
             h3("Maximum de Vraisemblance"),
             p("On remarque que Yi suit une loi binomiale ainsi la vraisemblance pour
                l'individu Yi est :"),
             withMathJax("$$L(y_i,\\beta)=p_i^{y_i}(1-p_i)^{1-y_i}$$"),
             p("Pour notre modèle nous aurons un echantillon ainsi sa
                log-vraisemblance sera :"),
             withMathJax("$$ln L(y,\\beta)=\\sum_{i=1}^N {y_iln[F(x_i\\beta)]+(1-y_i)ln[1-F(x_i\\beta)]}$$"),
             p("Nous allons devoir maintenant maximiser cette fonction afin d'avoir les
                coéfficients Betaqui maximisent la vraisemblance. Pour ce faire
                nous devons vérifier la condition du premièr et seconde ordre à l'aide
                du gradient et de la hessienne."),
             withMathJax("$$G(\\beta) = \\frac{\\partial \\log L(y, \\beta)}{\\partial \\beta} = \\sum_{i=1}^N \\frac{\\left[y_i - F\\left(x_i \\beta\\right)\\right] f\\left(x_i \\beta\\right)}{F\\left(x_i \\beta\\right) \\left[ 1 - F\\left(x_i \\beta\\right) \\right]} x_i^{\\prime}$$"),
             withMathJax("$$\\begin{aligned}\\underset{(K, K)}{H(\\beta)} = &  \\frac{\\partial^2 \\log L(y, \\beta)}{\\partial \\beta \\partial \\beta^{\\prime}} = - \\sum_{i=1}^N \\left[ \\frac{y_i}{F\\left(x_i \\beta\\right)^2} + \\frac{1 - y_i}{\\left[1 - F\\left(x_i \\beta\\right)\\right]^2} \\right] \\\\& f\\left(x_i \\beta\\right)^2 x_i^{\\prime} x_i + \\sum_{i=1}^N \\left[ \\frac{y_i - F\\left(x_i \\beta\\right)}{F\\left(x_i \\beta\\right) \\left[1 - F\\left(x_i \\beta\\right)\\right]} \\right] f^{\\prime}\\left(x_i \\beta\\right) x_i^{\\prime} x_i\\end{aligned}$$"),
             p("Nous pouvons maintenant extraire les Beta :"),
             withMathJax("$$
                        \\begin{gathered}
                        \\widehat{\\beta} = \\underset{\\{\\beta\\}}{\\arg \\max } [\\log L(y, \\beta)] \\\\
                        \\Longleftrightarrow \\frac{\\partial \\log L(y, \\widehat{\\beta})}{\\partial \\widehat{\\beta}} = \\sum_{i=1}^N \\frac{\\left[ y_i - F\\left( x_i \\widehat{\\beta} \\right) \\right] f\\left( x_i \\widehat{\\beta} \\right)}{F\\left( x_i \\widehat{\\beta} \\right) \\left[ 1 - F\\left( x_i \\widehat{\\beta} \\right) \\right]} x_i^{\\prime} \\\\
                        = G(\\widehat{\\beta}) = 0
                        \\end{gathered}
                        $$"),
             p("Il n'y a pas de solution analytique à ce problème c'est pourquoi il faut
                utiliser une méthode d'optimisation numérique. J'ai réalisé une video
                educative à ce sujet en compagnie de Romain Canelle : ",
                a(href = "https://youtu.be/q0s8yvuN75E", "Lien de la video")),
             ),
             column(4,
             h3("Interpretations"),
             p("Une fois le modèle construit nous obtenons les Beta et pouvons
                interprêter tout un tas de choses à partir de ceux-ci si tenté qu'ils
                soient significatifs.
                
                Pour ces interpretations, je vais m'aider des travaux dirigés de
                Mr.Guene réalisant le TD du Cours cité precedemment à Orléans."),
             br(),
             h4("Coefficients"),
             p("Nous pouvons seulement interprêter le signe des coefficients $\beta$ et
                non leur valeur.
                Si B1 = 1,5 alors une augmentation de la variable 1 induit une
                augmentation de la probabilité de connaître l'évènement."),
             br(),
             h4("La Côte"),
             p("La côte d'un individu i correspond son
                ratio de probabilité de connaitre l'évènement que de ne le connaîte.
                Si ci = 2 alors l'individu i a 2 fois plus de chance de connaître
                l'évènement que de ne le connaître."),
             withMathJax("$$c_i = \\frac{\\pi_i}{1-\\pi_i}$$"),
             h4("Rapport de côte"),
             p("Avec c'i la nouvelle côte. Si
                RCj = 1,3 alors si la variable j augmente de une unité alors la
                nouvelle côte sera de 1,3 fois l'ancienne."),
             withMathJax("$$RC_j = \\frac{c'_i}{c_i}$$"),
             h4("Effet marginal"),
             p("Si
                EMij = 0,05 alors une augmentation de 1 unité de la variable j
                pour l'individu i augmentera de 0,05 sa probabilité de connaître
                l'évènement."),
             withMathJax("$$EM_{i,j} = \\frac{\\partial \\pi_i}{\\partial x_{i,j}}$$"),
             )
            )
      ),
      tabPanel("Evaluation",
               div(style = "text-align: center;",
               p("Nous avons différents outils nous permettant d'evaluer un modèle comme
                  TAUa ou le D de sommers ainsi que l'AUC et la courbe ROC. Nous
                  verrons dans ce projet la courbe ROC et l'AUC (area under curve (courbe
                  ROC))."),
                  
                  p("Avant de construire tous ces indicateurs plusieurs choses sont
                  primordiales. Nous devons séparer nos données en 2 échantillons train
                  et test. La littérature s'accorde en général pour couper en 80%/20%
                  bien que ce ne soit pas le plus optimal tout le temps.
                  Nous estimons donc notre modèle sur l'échantillon train et l'évaluons
                  sur le test par la suite."),
                  
                  p("Si notre modèle fait bien les choses alors les probabilités éstimées
                  pour les individus ayant connu l'évènement devraient être supérieures.")),
                hr(),
               fluidRow(div(style = "text-align: center;",
                 column(6,
                        p("Lorsque l'on a choisit un seuil gamma et qu'ensuite on a obtenu nos
                          prédictions sur l'echantillon test, on peut cconstruire un tableau de
                          contingence avec les predictions et la vraie valeur de la variable
                          d'interêt."),
                        p("Soit (i;j) un couple d'individu pour qui i a connu l'évènement et
                          j nan. Alors pour ce couple/paire, si pi_i > pi_j alors c'est
                          une paire concordante, si pi_i < pi_j alors la paire est
                          discordante et si pi_i = pi_j alors la paire est liée. On notera
                          t le nombre de paire dans l'echantillon qui est égal au nombre de
                          personne ayant connu l'évènement fois le nombre de personne ne l'ayant
                          pas connu, n_c le nombre de paire concordante, n_d etc...
                          
                          L'AUC correspond à l'aire sous la courbe ROC ainsi puisque la courbe est
                          contenu sur un repère orthonormé allant de 0 à 1 pour l'axe des
                          abscisses et ordonnées alors un modèle aléatoire aura un AUC de 0.5 et
                          un modèle parfait aura un AUC de 1.
                          
                          La formule de l'AUC est la suivante :"),
                        withMathJax("$$ AUC = \\frac{(n_c + 0,5(t-n_c-n_d))}{t} $$"),
                        imageOutput("contin")
                        
                        ),
                 column(6,
                        p("A partir du tableau de contingence, nous pouvons définir la Spécificité et la
                        Sensitivité :"),
                        p("La spécificité correspond aux personnes n'ayant pas connu
                            l'évènement correctement identifiés par le modèle."),
                        p("La sensitivité correspond aux personnes ayant pas connu lm'évènement
                            correctement identifiés par le modèle."),
                        
                        p("On fait ceci pour des seuils gamma allant de 0 à 1 et après on peut
                        contruire la Courbe ROC.
                        Voici à quoi elle ressemble :"),
                        imageOutput("rocc")
                        )
                 
                 
               ))
               
               )
    ),
    
    tabPanel(title = "Exploration des données",
             sidebarLayout(
               
               sidebarPanel(width = 4,
                            tags$div(
                              style = "overflow-x: auto; width: 100%;",
                              DTOutput("base")
                            )
                  ),
               
               
               mainPanel(
                 tabsetPanel(
                   tabPanel("General", 
                            h3("Vue Globale"),
                            fluidRow(column(6,
                            p("Nos données proviennent de Taiwan. Elles datent d'Avril à Septembre 2005 
                              et contiennent des informations sur 30000 clients. On retrouve dans cette base
                              25 variables dont 23 explicatives."),
                            h5("Variable d'interêt :"),
                            tags$ul(
                              tags$li(
                                p(em("Défaut"),": 1 si le client a fait défaut, 0 sinon.")
                              )),
                            h5("Variables qualitatives :"),
                            tags$ul(
                              tags$li(
                                p(em("SEX"),": Représente le sexe de l'individu, 1 si homme, 2 si femme.")
                              ),
                              tags$li(
                                p(em("EDUCATION"),": Représente le niveau d'étude d'un individu. Nous avons '1'
                                   étude supérieures, '2' université, '3' Lycée, '4' autres.")
                              ),
                              tags$li(
                                p(em("MARRIAGE"),": Etat civil de l'individu. Nous avons '1'
                                   Marié, '2' celibataire, '3' Divorcé, '4' autres.")
                              ),
                              tags$li(
                                p(em("PAY_1...6"),": Représente les statuts de remboursement au cours des 6 derniers mois.
                                  Nous avons '-2' Pas de consommation, '-1' Payé en totalité, '0' Utilisation de crédit renouvelable
                                  , '1' à '8' Retard de paiement de 1 à 8 mois.")
                              ),
                            ),
                            h5("Variables quantitatives :"),
                            tags$ul(
                              tags$li(
                                p(em("BILL_AMT1...6"),": Montant des factures chaque mois.")
                              ),
                              tags$li(
                                p(em("PAY_AMT1...6"),": Montant payé chaque mois.")
                              ),
                            ),
                            
                            ),
                            column(6,
                                   div(style = "text-align: center;",
                                   tableOutput("prop_def"),
                                   plotOutput(("prop_defg"))
                                   )
                                   ))),
                   tabPanel("Montant du crédit", 
                            plotOutput("montcredg"),
                            tableOutput("montcred"),
                            p("Les clients ne faisant pas défaut ont en moyenne des crédits plus grands
                              que ceux faisant défaut. Cela peut s'expliquer par le fait que la banque
                              avait de base des suspicions et n'a pas accoré un grand crédit mais aussi
                              par le fait que les personnes ayant des crédits volumineux sont des personnes
                              ayant plus de moyens et donc peut être plus a même de pouvoir rembourser.")
                            ),
                   tabPanel("Sexe", 
                            plotOutput("sexeg"),
                            tableOutput("sexe"),
                            p("Nous retrouvons bien plus de femmes que d'hommes.")
                   ),
                   tabPanel("Age", 
                            plotOutput("ageg"),
                            tableOutput("age"),
                            p("Les répartitions sont les mêmes si ce n'est que l'on peut
                              remarquer que les femmes sont légèrement plus agées.")
                   ),
                   tabPanel("Education", 
                            plotOutput("educationg"),
                            tableOutput("education"),
                            p("On remarque que les personnes allant à l'université font plus souvent défaut
                              que les autres.")
                   ),
                   tabPanel("Marriage", 
                            plotOutput("marriageg"),
                            tableOutput("marriage"),
                            p("C'est assez homogène, même si l'on dénote que les personnes marriés 
                              font défaut un tout petit peu plus souvent que les autres.")
                   ),
                   tabPanel("PAY_", 
                            tableOutput("pay"),
                            p("On remarque que les personnes allant à l'université font plus souvent défaut
                              que les autres.")
                   ),
                   tabPanel("BILL_AMT", 
                            plotOutput("billg"),
                            tableOutput("bill"),
                            p("C'est assez homogène, même si l'on dénote que les personnes ne faisant pas défaut ont
                              factures plus élevées que les autres.")
                   ),
                   tabPanel("PAY_AMT", 
                            plotOutput("payag"),
                            tableOutput("paya"),
                            p("On remarque que les personnes les plus à même de faire défaut sont celles
                              ayant des retards de paiement, rien de surprenant ici.")
                   ),
                   tabPanel("Correlation", 
                            plotOutput("corr"),
                            plotOutput("corr2"),
                            plotOutput("corr3"),
                            plotOutput("corr4")
                            )
                 )
               )
             )
             
             ),
    navbarMenu("Analyse",
               tabPanel("Modélisation",
                        sidebarLayout(
                          
                          
                          sidebarPanel(
                            radioButtons("modelec", "Modèle souhaité :",
                                         choices = c("Logit entier" = "logit", "Probit entier" = "probit"
                                                     ,"Logit BIC"= "logitbic", "Probit BIC"= "probitbic")),
                            radioButtons("inter", "Interprétations souhaitées :",
                                         choices = c("Variables quantitatives" = "quant", "Variables qualitatives" = "qual")),
                            h5("Variable d'interêt :"),
                            tags$ul(
                              tags$li(
                                p(em("Défaut"),": 1 si le client a fait défaut, 0 sinon.")
                              )),
                            h5("Variables qualitatives :"),
                            tags$ul(
                              tags$li(
                                p(em("SEX"),": Représente le sexe de l'individu, 1 si homme, 2 si femme.")
                              ),
                              tags$li(
                                p(em("EDUCATION"),": Représente le niveau d'étude d'un individu. Nous avons '1'
                                   étude supérieures, '2' université, '3' Lycée, '4' autres.")
                              ),
                              tags$li(
                                p(em("MARRIAGE"),": Etat civil de l'individu. Nous avons '1'
                                   Marié, '2' celibataire, '3' Divorcé, '4' autres.")
                              ),
                              tags$li(
                                p(em("PAY_1...6"),": Représente les statuts de remboursement au cours des 6 derniers mois.
                                  Nous avons '-2' Pas de consommation, '-1' Payé en totalité, '0' Utilisation de crédit renouvelable
                                  , '1' à '8' Retard de paiement de 1 à 8 mois.")
                              ),
                            ),
                            h5("Variables quantitatives :"),
                            tags$ul(
                              tags$li(
                                p(em("BILL_AMT1...6"),": Montant des factures chaque mois.")
                              ),
                              tags$li(
                                p(em("PAY_AMT1...6"),": Montant payé chaque mois.")
                              ),
                            )
                          ),
                          
                          
                          mainPanel(
                            tabsetPanel(
                              tabPanel("Coefficients",
                            fluidRow(
                              column(6,
                                     tags$div(
                                       style = "max-height: 800px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;",
                                     DTOutput("coef")),
                                     ),
                              column(6,
                                     h3("Interprétation"),
                                     textOutput("coeft"),
                                     )
                                  )
                              ),
                            tabPanel("Rapport de côte",
                                     fluidRow(
                                       column(6,
                                              imageOutput("rci")
                                              
                                       ),
                                       column(6,
                                              h3("Interpretation"),
                                              textOutput("rct"),
                                              imageOutput("rci2")
                                       )
                                     )
                                     )
                          ) 
                          )
                        )
                        
                        
                        
                        
                        
                        
                        ),
               tabPanel("Evaluation",
                        sidebarLayout(div(style = "text-align: center;",
                          sidebarPanel(width = 4,
                            radioButtons("bic", "Type de Modéle:",
                                         choices = c("Entier" = "entier", "Minimisant critère BIC" = "bic")),
                            h3("Matrice de confusion"),
                            sliderInput("gammal", label = "Seuil gamma demandé Logit:",
                                        min = 0, max = 1, value = 0.49, step = 0.01),
                            tableOutput("lerreur"),
                            sliderInput("gammap", label = "Seuil gamma demandé Probit:",
                                        min = 0, max = 1, value = 0.49, step = 0.01),
                            tableOutput("perreur")
                          )),
                        mainPanel(
                          fluidRow(
                            column(6,
                                   div(style = "height: 700px; overflow-y: auto; border: 1px solid lightgray; padding: 10px",
                              plotOutput("lroc"),
                              verbatimTextOutput("lauc"),
                              DTOutput("lpred"))
                            ),
                            column(6,
                                   div(style = "height: 700px; overflow-y: auto; border: 1px solid lightgray; padding: 10px",
                              plotOutput("proc"),
                              verbatimTextOutput("pauc"),
                              DTOutput("ppred"))
                            )
                          )
                          )
                        )
                )
    ),
    
    tabPanel("Prédictions",
                 p("Vous pouvez desormais sur la base de nos modèles, un quelque peu fragiles par ailleurs,
                   prédire si un individu est susceptible de faire défaut ou non."),
                 fluidRow(
                   div(style = "text-align: center;",
                   column(3,
                          h5("Informations personnelles"),
                          textInput("rnom","Nom",value = "Lacroix Ewan"),
                          selectInput("rsex", "Sexe :", choices = c("Homme" = 1,"Femme" = 2)),
                          selectInput("reduc", "Niveau d'education",choices = c(
                            "Etudes superieures" = 1, "Université"=2, "Lycée"=3,
                            "Autres" = 4)),
                          selectInput("rmarriage", "Statut marital",choices = c(
                            "Marié" = 1, "Celibataire"=2, "Divorcé"=3,
                            "Autres" = 4)),
                          sliderInput("rage","Age", value = 30,min = 18,max = 80,step = 1),
                          numericInput("montant","Montant Crédit :",value = 150000,
                                       min=0, max = 1000000,step = 1000)
                   ),
                   column(3,
                          h5("Montant des factures chaque mois"),
                          numericInput("b1","Montant facture mois-1",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("b2","Montant facture mois-2",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("b3","Montant facture mois-3",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("b4","Montant facture mois-4",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("b5","Montant facture mois-5",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("b6","Montant facture mois-6",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                    ),
                   column(3,
                          h5("Montant payé chaque mois"),
                          numericInput("m1","Montant payé mois-1",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("m2","Montant payé mois-2",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("m3","Montant payé mois-3",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("m4","Montant payé mois-4",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("m5","Montant payé mois-5",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                          numericInput("m6","Montant payé mois-6",value = 10000,
                                       min = -10000,max = 100000,step = 1),
                    ),
                   column(3,
                          h5("Statut remboursement chaque mois"),
                          numericInput("s1","Statut mois-1",value = -2,
                                       min = -2,max = 7,step = 1),
                          numericInput("s2","Statut mois-2",value = -2,
                                       min = -2,max = 7,step = 1),
                          numericInput("s3","Statut mois-3",value = -2,
                                       min = -2,max = 7,step = 1),
                          numericInput("s4","Statut mois-4",value = -2,
                                       min = -2,max = 7,step = 1),
                          numericInput("s5","Statut mois-5",value = -2,
                                       min = -2,max = 7,step = 1),
                          numericInput("s6","Statut mois-6",value = -2,
                                       min = -2,max = 7,step = 1),
                          ))
                 ),
                fluidRow(
                  column(4,
                         radioButtons("rchoix", "Type de Modéle:",
                                      choices = c("Logit entier" = "logit", "Probit entier" = "probit"
                                                  ,"Logit BIC"= "logitbic", "Probit BIC"= "probitbic"))
                         ),
                  column(8,
                         tags$div(
                           style = "background-color: #e6f2ff; padding: 15px; border-radius: 10px; border: 1px solid #b3d7ff; margin-bottom: 20px;",
                         div(style = "text-align: center;",
                         textOutput("result")))
                         )
                )
                


             
    )
  )
)



server <- function(input, output) {
  
  df = read_csv("UCI_Credit_Card.csv")
  df2 = cor(read_csv("UCI_Credit_Card.csv")[,-1])
  df= df[2:ncol(df)]
  
  df$SEX = as.factor(df$SEX)
  
  df$default.payment.next.month = as.factor(df$default.payment.next.month)
  df = rename(df,defaut = default.payment.next.month)
  
  df = mutate(df,EDUCATION = ifelse(EDUCATION <1,4,EDUCATION))
  df = mutate(df,EDUCATION = ifelse(EDUCATION >3,4,EDUCATION))
  df$EDUCATION = as.factor(df$EDUCATION)
  
  df = mutate(df,MARRIAGE = ifelse(MARRIAGE <1,4,MARRIAGE))
  df$MARRIAGE = as.factor(df$MARRIAGE)
  
  df$PAY_0 = as.factor(df$PAY_0)
  df$PAY_2 = as.factor(df$PAY_2)
  df$PAY_3 = as.factor(df$PAY_3)
  df$PAY_4 = as.factor(df$PAY_4)
  df$PAY_5 = as.factor(df$PAY_5)
  df$PAY_6 = as.factor(df$PAY_6)
  df = rename(df,PAY_1 = PAY_0)
  
  set.seed(123)
  echantillon = sample(nrow(df),round(0.8*nrow(df)))
  
  train = df[echantillon,]
  test = df[-echantillon,]
  
  modele = glm(defaut~.,train,family = binomial(link = "logit"))
  modele2 = glm(formula = defaut ~ LIMIT_BAL + MARRIAGE+ SEX + EDUCATION + AGE + PAY_1 + BILL_AMT1+ BILL_AMT2 + PAY_AMT1 + PAY_AMT2, family = binomial(link = "logit"), data = train)
  modelep = glm(defaut~.,train,family = binomial(link = "probit"))
  modelep2 = glm(formula = defaut ~ LIMIT_BAL + MARRIAGE+ SEX + EDUCATION + AGE + PAY_1 + BILL_AMT1+ BILL_AMT2 + PAY_AMT1 + PAY_AMT2, family = binomial(link = "probit"), data = train)
  
  pred = predict(modele, newdata = test, type = "response")
  pred2 = predict(modele2, newdata = test, type = "response")
  predp = predict(modelep, newdata = test, type = "response")
  predp2 = predict(modelep2, newdata = test, type = "response")
  
  
  
  output$base <- renderDT({
    datatable(df,options = list(pageLength = 13))
  })
  
  output$montcred = renderTable({
    summarise(group_by(df,SEX,defaut),Montant_credit = mean(LIMIT_BAL))
  })
  
  output$montcredg = renderPlot({
    ggplot(df,aes(interaction(defaut,SEX), LIMIT_BAL)) + geom_boxplot(fill = "azure") + labs(title = "Montant du crédit accordé", x = "Defaut/Sexe", y = "Montant") + theme_minimal()
  })
  
  output$prop_def = renderTable({
    summarise(group_by(df,defaut), Nombre = n(), Proportion = Nombre/nrow(df))
  })
  
  output$prop_defg = renderPlot({
    ggplot(summarise(group_by(df,defaut), Nombre = n(), Proportion = Nombre/nrow(df)),aes(x = defaut,y= Nombre)) + geom_bar(stat = "identity") + labs(title = "Proportion d'individus ayant fait défaut", x = "Defaut", y = "Nombre") + theme_minimal()
  })
  
  output$corr = renderPlot({
    corrplot(cor(df2[,c(12,13,14,15,16,17)]), method = "number")
    })
  
  output$corr2 = renderPlot({
    corrplot(cor(df2[,c(6,7,8,9,10,11)]), method = "number")
  })
  
  output$corr3 = renderPlot({
    corrplot(cor(df2[,c(18,19,20,21,22,23)]), method = "number")
  })
  
  output$corr4 = renderPlot({
    corrplot(cor(df2[,c(2,3,4,1,5,24)]), method = "number")
  })
  
  output$sexe = renderTable({
    summarise(group_by(df,SEX,defaut), Nombre = n(), Proportion = Nombre/30000)
  })
  
  output$sexeg = renderPlot({
    ggplot(summarise(group_by(df,SEX,defaut), Nombre = n()),aes(x = interaction(SEX,defaut),y= Nombre)) + geom_bar(stat = "identity") + labs(title = "Répartition Sexe", x = "Sexe/defaut", y = "Nombre") + theme_minimal()
  })
  
  output$age = renderTable({
    summarise(group_by(df,SEX,defaut), Age_moyen = mean(AGE))
  })
  
  output$ageg = renderPlot({
    ggplot(df,aes(interaction(defaut,SEX), AGE)) + geom_boxplot(fill = "azure") + labs(title = "Age Moyen", x = "Defaut/Sexe", y = "Montant") + theme_minimal()
  })
  
  output$education = renderTable({
    summarise(group_by(df,EDUCATION,defaut), Nombre = n(), Proportion = Nombre/30000)
  })
  
  output$educationg = renderPlot({
    ggplot(summarise(group_by(df,EDUCATION,defaut), Nombre = n()),aes(x = interaction(EDUCATION,defaut),y= Nombre)) + geom_bar(stat = "identity") + labs(title = "Répartition Niveau d'education", x = "Education/defaut", y = "Nombre") + theme_minimal()
  })
  
  output$marriage = renderTable({
    summarise(group_by(df,MARRIAGE,defaut), Nombre = n(), Proportion = Nombre/30000)
  })
  
  output$marriageg = renderPlot({
    ggplot(summarise(group_by(df,MARRIAGE,defaut), Nombre = n()),aes(x = interaction(MARRIAGE,defaut),y= Nombre)) + geom_bar(stat = "identity") + labs(title = "Répartition par statut marrital", x = "Statut/defaut", y = "Nombre") + theme_minimal()
  })

  output$paya = renderTable({
    summarise(group_by(df,defaut,SEX), Moyenne = mean(PAY_AMT1))
  })
  
  output$payag = renderPlot({
    ggplot(df,aes(interaction(defaut,SEX), PAY_AMT1)) + geom_boxplot(fill = "azure") + labs(title = "Montant payé Moyen", x = "Defaut/Sexe", y = "Montant") + theme_minimal()
  })
  
  output$bill = renderTable({
    summarise(group_by(df,defaut,SEX), Moyenne = mean(BILL_AMT1))
  })
  
  output$billg = renderPlot({
    ggplot(df,aes(interaction(defaut,SEX), BILL_AMT1)) + geom_boxplot(fill = "azure") + labs(title = "Montant payé Moyen", x = "Defaut/Sexe", y = "Montant") + theme_minimal()
  })
  
  output$pay = renderTable({
    summarise(group_by(df,defaut,PAY_1), Nombre = n())
  })
  
  output$coef = renderDT({
  
    if (input$modelec == "logit") {
      lala = summary(modele)
      lala =data.frame(lala$coefficients)[,c(-2,-3)]
      lala = rename(lala,"P-value"= "Pr...z..")
      datatable(mutate_if(lala,is.numeric, ~ round(., 7)),options = list(pageLength = 14))
    } else if (input$modelec == "logitbic") {
      lala2 = summary(modele2)
      lala2 =data.frame(lala2$coefficients)[,c(-2,-3)]
      lala2 = rename(lala2,"P-value"= "Pr...z..")
      datatable(mutate_if(lala2,is.numeric, ~ round(., 7)),options = list(pageLength = 14))
    } else if (input$modelec == "probitbic") {
      lala3 = summary(modelep2)
      lala3 =data.frame(lala3$coefficients)[,c(-2,-3)]
      lala3 = rename(lala3,"P-value"= "Pr...z..")
      datatable(mutate_if(lala3,is.numeric, ~ round(., 7)),options = list(pageLength = 14))
    } else if (input$modelec == "probit") {
      lala4 = summary(modelep)
      lala4 =data.frame(lala4$coefficients)[,c(-2,-3)]
      lala4 = rename(lala4,"P-value"= "Pr...z..")
      datatable(mutate_if(lala4,is.numeric, ~ round(., 7)),options = list(pageLength = 14))
    } 
  })
  
  output$rci = renderImage({
      if (input$modelec == "logit") {
        list(
          src = "www/rc1.png",
          width = 350,
          height = 1200,
          alt = "RC"
        )

      } else if (input$modelec == "logitbic") {
        list(
          src = "www/rc2.png",
          width = 450,
          height = 800,
          alt = "RC"
        )
      } else if (input$modelec == "probitbic") {
        list(
          src = "www/rcp2.png",
          width = 450,
          height = 800,
          alt = "RC"
        )

      } else if (input$modelec == "probit") {
        list(
          src = "www/testo.png",
          width = 350,
          height = 1200,
          alt = "RC"
        )

      }

  },deleteFile = FALSE)
  
  output$rci2 = renderImage({
    if (input$modelec == "logit") {
      list(
        src = "www/rc11.png",
        width = 350,
        height = 1200,
        alt = "RC"
      )
      
    } else if (input$modelec == "logitbic") {
      list(
        src = "www/bic.png",
        width = 350,
        height = 100,
        alt = "RC"
      )
    } else if (input$modelec == "probitbic") {
      list(
        src = "www/bic.png",
        width = 350,
        height = 100,
        alt = "RC"
      )
      
    } else if (input$modelec == "probit") {
      list(
        src = "www/rcp11.png",
        width = 350,
        height = 1200,
        alt = "RC"
      )
      
    }
    
  },deleteFile = FALSE)
  
  
  output$coeft = renderText({
    if (input$inter == "quant") {
    if (input$modelec == "logit") {
      paste("Il est difficile de le voir ici, mais le coefficient associé à LIMIT_BAL est négatif, ainsi 
            plus le montant du crédit est important et moins l'on a de chance de faire défaut, cela peut
            paraître contre intuitif notamment avec les résultats vu précedemment mais on peut voir ça 
            comme le fait que ce sont des personnes potentiellement plus aisés ou digne de rembourser
            d'ou l'octroie de crédit important. On remarque aussi que plus l'âge augmente est plus la probabilité 
            de faire défaut augmente. Concernant les montant des factures, on remarque qu'ils ne sont pas significatifs 
            on s'en doutait un peu au vu des corrélations et statistiques descriptives qui ne nous donnaient pas 
            vraiment d'information. Cependant le montant remboursé au cours des 2 derniers mois
            est un bon indicateur puisque leurs
            coefficients sont négatifs ainsi plus le montant payé chaque mois est grand et moins l'individu a de chance
            de faire défaut ce qui semble logique." )
    } else if (input$modelec == "logitbic") {
      paste("On remarque encore une fois que plus le montant du crédit accordé est élevé est plus la chance de 
            faire defaut baisse. On retrouve le même mecanisque pour le montant de la facture du mois dernier qui 
            peut être mis en correlation avec nos analyses precedentes puisque peut paraître contre intuitif
            de premier abord. On retrouve ensuite les interpretations classiques pour le montant payé des 2 derniers mois. Il
            en va de même pour l'âge qui augmente les chance de faire défaut avec le temps.")
    } else if (input$modelec == "probitbic") {
      paste("On remarque encore une fois que plus le montant du crédit accordé est élevé est plus la chance de 
            faire defaut baisse. On retrouve le même mecanisque pour le montant de la facture du mois dernier qui 
            peut être mis en correlation avec nos analyses precedentes puisque peut paraître contre intuitif
            de premier abord, on remarque que sa p-value est legerement supérieure à 5% on se propose de le garder.
            . On retrouve ensuite les interpretations classiques pour le montant payé des 2 derniers mois. Il
            en va de même pour l'âge qui augmente les chance de faire défaut avec le temps.")
    } else if (input$modelec == "probit") {
      paste("Les analyses sont les mêmes que pour le Logit, nos résultats sont sensiblement les mêmes, 
            le coefficient associé à LIMIT_BAL est négatif, ainsi 
            plus le montant du crédit est important et moins l'on a de chance de faire défaut, cela peut
            paraître contre intuitif notamment avec les résultats vu précedemment mais on peut voir ça 
            comme le fait que ce sont des personnes potentiellement plus aisés ou digne de rembourser
            d'ou l'octroie de crédit important. On remarque aussi que plus l'âge augmente est plus la probabilité 
            de faire défaut augmente. Concernant les montant des factures, on remarque qu'ils ne sont pas significatifs 
            on s'en doutait un peu au vu des corrélations et statistiques descriptives qui ne nous donnaient pas 
            vraiment d'information. Cependant le montant remboursé au cours des 2 derniers mois
            est un bon indicateur puisque leurs
            coefficients sont négatifs ainsi plus le montant payé chaque mois est grand et moins l'individu a de chance
            de faire défaut ce qui semble logique.")
    } } else {
      if (input$modelec == "logit") {
        paste("On remarque ici, qu'être une femme baisse les chances d'être défaut comparé à être un homme.
              Aussi, le niveau d'étude ne semble pas significatif. Etre celibataire baisse les chances de faire
              defaut comparé à être marié. Enfin, concernant le statut de remboursement des mois derniers, on remarque
              que plus l'on est dans un statut élevé et donc que l'individu met plusieurs mois à rembourser par 
              rapport à quelqu'un qui n'a pas de consommation, et bien il aura de chance de faire défaut. Evidemment,
              si l'on a du mal à rembourser alors on a plus de chance de faire defaut." )
      } else if (input$modelec == "logitbic") {
        paste("Nous avons les mêmes analyse que dans le Logit, être une femme baisse les chances d'être défaut comparé à être un homme.
              Aussi, le niveau d'étude ne semble pas significatif. Etre celibataire baisse les chances de faire
              defaut comparé à être marié. Enfin, concernant le statut de remboursement des mois derniers, on remarque
              que plus l'on est dans un statut élevé et donc que l'individu met plusieurs mois à rembourser par 
              rapport à quelqu'un qui n'a pas de consommation, et bien il aura de chance de faire défaut. Evidemment,
              si l'on a du mal à rembourser alors on a plus de chance de faire defaut.")
      } else if (input$modelec == "probitbic") {
        paste("Nous avons les mêmes analyse que dans le Logit, être une femme baisse les chances d'être défaut comparé à être un homme.
              Aussi, le niveau d'étude ne semble pas significatif. Etre celibataire baisse les chances de faire
              defaut comparé à être marié. Enfin, concernant le statut de remboursement des mois derniers, on remarque
              que plus l'on est dans un statut élevé et donc que l'individu met plusieurs mois à rembourser par 
              rapport à quelqu'un qui n'a pas de consommation, et bien il aura de chance de faire défaut. Evidemment,
              si l'on a du mal à rembourser alors on a plus de chance de faire defaut.")
      } else if (input$modelec == "probit") {
        paste("Nous avons les mêmes analyse que dans le Logit, être une femme baisse les chances d'être défaut comparé à être un homme.
              Aussi, le niveau d'étude ne semble pas significatif. Etre celibataire baisse les chances de faire
              defaut comparé à être marié. Enfin, concernant le statut de remboursement des mois derniers, on remarque
              que plus l'on est dans un statut élevé et donc que l'individu met plusieurs mois à rembourser par 
              rapport à quelqu'un qui n'a pas de consommation, et bien il aura de chance de faire défaut. Evidemment,
              si l'on a du mal à rembourser alors on a plus de chance de faire defaut.")
      }
    }
  })
  
  output$rct = renderText({
    if (input$inter == "quant") {
    if (input$modelec == "logit") {
      paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. Nous remarquons aucun effet significatif sur le rapport de chance
            concernant les variables auantitatives si ce n'est l'âge qui lorsqu'il augmente d'une unité, augmente 
            notre rapport de chance initial de 0,4%.")
    } else if (input$modelec == "logitbic") {
      paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. Augmenter le montant de credit d'une unité baissera notrre rapport 
            de chance de faire defaut de 0,0003% ce qui est assez élevé contenu du fait que les montants sont élevés.
            Il en va de même pour le montant des factures du mois précedent. 
            L'âge elle l'augmente de 0,4% et le montant de la facture de l'avant dernier mois de 0;0006%.
            Augmenter le montant payé du mois precedent d'une unité baisse notre rapport de chance de faire défaut
            de 0,0018% ce qui rentre en correlation avec nos analyses précedentes.")
    } else if (input$modelec == "probitbic") {
      paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. On retrouve ici des analyses assez similaires à celles du logit BIC 
            mais nous ne pouvons le voir clairement de par l'arrondissement.")
    } else if (input$modelec == "probit") {
      paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite.Nous remarquons aucun effet significatif sur le rapport de chance
            concernant les variables auantitatives si ce n'est l'âge qui lorsqu'il augmente d'une unité, augmente 
            notre rapport de chance initial de 0,2%.")
    } } else {
      
      if (input$modelec == "logit") {
        paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite.
              Passer du statut d'homme à femme baisse notre rapport de chance de faire défaut de 14%.
              Un individu etant allé maximum à l'université verra son rapport de chance multiplié par 1,026
              comparé à un individu ayant fait des études supérieures.
              Une personne celibataire verra son rapport de chance baisser de 14% comparé à une personne mariée.
              Concernant les statuts de remboursements, plus l'individu passe d'un statut de non consommation à mettre
              1 ou plusieurs mois à rembourser, plus son rapport de chance augmente.")
      } else if (input$modelec == "logitbic") {
        paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. Passer du statut d'homme à femme baisse notre rapport de chance de faire défaut de 16%.
              Un individu etant allé maximum à l'université verra son rapport de chance multiplié par 1,028
              comparé à un individu ayant fait des études supérieures.
              Une personne celibataire verra son rapport de chance baisser de 14% comparé à une personne mariée.
              Concernant les statuts de remboursements, plus l'individu passe d'un statut de non consommation à mettre
              1 ou plusieurs mois à rembourser, plus son rapport de chance augmente.")
      } else if (input$modelec == "probitbic") {
        paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. Passer du statut d'homme à femme baisse notre rapport de chance de faire défaut de 9%.
              Un individu etant allé maximum à l'université verra son rapport de chance multiplié par 1,022
              comparé à un individu ayant fait des études supérieures.
              Une personne celibataire verra son rapport de chance baisser de 8% comparé à une personne mariée.
              Concernant les statuts de remboursements, plus l'individu passe d'un statut de non consommation à mettre
              1 ou plusieurs mois à rembourser, plus son rapport de chance augmente.")
      } else if (input$modelec == "probit") {
        paste("Pour des raisons de temps de calcul pour les intervalles, je l'ai calculé en amont et je mets une image
            des résultats pour aller plus vite. Passer du statut d'homme à femme baisse notre rapport de chance de faire défaut de 8%.
              Un individu etant allé maximum à l'université verra son rapport de chance multiplié par 1,023
              comparé à un individu ayant fait des études supérieures.
              Une personne celibataire verra son rapport de chance baisser de 8% comparé à une personne mariée.
              Concernant les statuts de remboursements, plus l'individu passe d'un statut de non consommation à mettre
              1 ou plusieurs mois à rembourser, plus son rapport de chance augmente.")
      }
      
    }
  })
  
  output$lerreur = renderTable({
    if (input$bic == "entier"){
    table(Prediction = ifelse(pred > input$gammal, 1, 0), Réel = test$defaut)
    } else {
    table(Prediction = ifelse(pred2 > input$gammal, 1, 0), Réel = test$defaut)
    }
  })
  
  output$perreur = renderTable({
    if (input$bic == "entier"){
      table(Prediction = ifelse(predp > input$gammap, 1, 0), Réel = test$defaut)
    } else {
      table(Prediction = ifelse(predp2 > input$gammap, 1, 0), Réel = test$defaut)
    }
  })
  
  output$lroc = renderPlot({
    if (input$bic == "entier"){
      plot(roc(test$defaut, pred), main = "Courbe ROC Logit", col = "black", ylab = "Sensitivité", xlab="Spécificité")
    } else {
      plot(roc(test$defaut, pred2), main = "Courbe ROC Logit", col = "black", ylab = "Sensitivité", xlab="Spécificité")
    }
  })
  
  output$proc = renderPlot({
    if (input$bic == "entier"){
      plot(roc(test$defaut, predp), main = "Courbe ROC Probit", col = "black", ylab = "Sensitivité", xlab="Spécificité")
    } else {
      plot(roc(test$defaut, predp2), main = "Courbe ROC Probit", col = "black", ylab = "Sensitivité", xlab="Spécificité")
    }
  })
  
  output$lauc = renderPrint({
    if (input$bic == "entier"){
      auc(roc(test$defaut, pred))
    } else {
      auc(roc(test$defaut, pred2))
    }
  })
  
  output$pauc = renderPrint({
    if (input$bic == "entier"){
      auc(roc(test$defaut, predp))
    } else {
      auc(roc(test$defaut, predp2))
    }
  })
  
  output$lpred = renderDT({
    if (input$bic == "entier"){
      seuil = input$gammal
      yibar = ifelse(pred > seuil, 1, 0)
      apercu = data.frame("Defaut" = test$defaut,"Probabilité estimée"=pred,"yi_bar" = yibar)
      as.data.frame.matrix(apercu)
    } else {
      seuil = input$gammal
      yibar = ifelse(pred2 > seuil, 1, 0)
      apercu = data.frame("Defaut" = test$defaut,"Probabilité estimée"=pred,"yi_bar" = yibar)
      as.data.frame.matrix(apercu)
    }
  })

  output$ppred = renderDT({
    if (input$bic == "entier"){
      seuil = input$gammap
      yibar = ifelse(predp > seuil, 1, 0)
      apercu = data.frame("Defaut" = test$defaut,"Probabilité estimée"=pred,"yi_bar" = yibar)
      as.data.frame.matrix(apercu)
    } else {
      seuil = input$gammap
      yibar = ifelse(predp2 > seuil, 1, 0)
      apercu = data.frame("Defaut" = test$defaut,"Probabilité estimée"=pred,"yi_bar" = yibar)
      as.data.frame.matrix(apercu)
    }
  })
  
  output$result = renderText({
    toto = data.frame("LIMIT_BAL" =input$montant,"SEX" = as.factor(input$rsex), "EDUCATION" = as.factor(input$reduc),
                      "MARRIAGE"=as.factor(input$rmarriage),"AGE" = input$rage,"PAY_1" = as.factor(input$s1),
                      "PAY_2" = as.factor(input$s2),"PAY_3" = as.factor(input$s3),"PAY_4" = as.factor(input$s4),
                      "PAY_5" = as.factor(input$s5),"PAY_6" = as.factor(input$s6),
                      "BILL_AMT1" = input$b1,"BILL_AMT2" = input$b2,"BILL_AMT3" = input$b3,"BILL_AMT4" = input$b4,
                      "BILL_AMT5" = input$b5,"BILL_AMT6" = input$b6, "PAY_AMT1" =input$m1,"PAY_AMT2" =input$m2,
                      "PAY_AMT3" =input$m3,"PAY_AMT4" =input$m4,"PAY_AMT5" =input$m5,"PAY_AMT6" =input$m6)
    if (input$rchoix == "logit") {
      proba = predict(modele, newdata = toto, type = "response")
      scorelin = -log(proba**(-1)-1)
      cat = ifelse(proba<0.1,"A+++",ifelse(proba<=0.3,"A++",ifelse(proba<=0.5,"A+",
                        ifelse(proba<=0.7,"A-","A--"))))
      cote = proba/(1-proba)
      paste("Le score de",input$rnom, "est de", scorelin,", soit une probabilité
      de défaut de", proba,".", input$rnom, "est un très bon client", cat,".",input$rnom,"a",cote,"fois plus
            de chance de faire défaut que de ne pas le faire.")
    } else if (input$rchoix == "probit") {
      proba = predict(modelep, newdata = toto, type = "response")
      scorelin = -log(proba**(-1)-1)
      cat = ifelse(proba<0.1,"A+++",ifelse(proba<=0.3,"A++",ifelse(proba<=0.5,"A+",
                                                                   ifelse(proba<=0.7,"A-","A--"))))
      cote = proba/(1-proba)
      paste("Le score de",input$rnom, "est de", scorelin,", soit une probabilité
      de défaut de", proba,".", input$rnom, "est un très bon client", cat,".",input$rnom,"a",cote,"fois plus
            de chance de faire défaut que de ne pas le faire.")
    } else if (input$rchoix == "logitbic") {
      proba = predict(modele2, newdata = toto, type = "response")
      scorelin = -log(proba**(-1)-1)
      cat = ifelse(proba<0.1,"A+++",ifelse(proba<=0.3,"A++",ifelse(proba<=0.5,"A+",
                                                                   ifelse(proba<=0.7,"A-","A--"))))
      cote = proba/(1-proba)
      paste("Le score de",input$rnom, "est de", scorelin,", soit une probabilité
      de défaut de", proba,".", input$rnom, "est un très bon client", cat,".",input$rnom,"a",cote,"fois plus
            de chance de faire défaut que de ne pas le faire.")
    } else if (input$rchoix == "probitbic") {
      proba = predict(modelep2, newdata = toto, type = "response")
      scorelin = -log(proba**(-1)-1)
      cat = ifelse(proba<0.1,"A+++",ifelse(proba<=0.3,"A++",ifelse(proba<=0.5,"A+",
                                                                   ifelse(proba<=0.7,"A-","A--"))))
      cote = proba/(1-proba)
      paste("Le score de",input$rnom, "est de", scorelin,", soit une probabilité
      de défaut de", proba,".", input$rnom, "est un très bon client", cat,".",input$rnom,"a",cote,"fois plus
            de chance de faire défaut que de ne pas le faire.")
    }
    
  })
  
  output$contin = renderImage({
      list(
        src = "www/contingence.png",
        width = 250,
        height = 250,
        alt = "Table de contingence"
      )
    })
  
  output$rocc = renderImage({
    list(
      src = "www/courberoc.png",
      width = 500,
      height = 300,
      alt = "Courbe ROC"
    )
  })
  
}


shinyApp(ui = ui, server = server)
