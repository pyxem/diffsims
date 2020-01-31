# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

# Electron scattering factors from Lobato & Van Dyck (2014):
# "An accurate parameterization for scattering factors, electron densities and
# electrostatic potentials for neutral atoms that obey all physical constraints"

# name  a1, b1, a2, b2, a3, b3, a4, b4, a5, b5

ATOMIC_SCATTERING_PARAMS_LOBATO = dict(
    (('H',
      ((0.006473848488352918, 2.785198853791489),
       (-0.490192576780229, 2.776204283306448),
          (0.5732841603908765, 2.775385910506251),
          (-0.3794033014839905, 2.767593028672588),
          (0.5544264747740791, 2.765118976429275))),
        ('He',
         ((3.057451160998355, 1.089672487260788),
          (-62.00447791273253, 0.9398387981431211),
          (64.00555370846145, 0.9252890343862655),
          (-5.001325785427806, 0.8229474987086506),
          (0.1517988287005264, 0.5773931106754022))),
        ('Li',
         ((3.926222728861479, 8.142760135172804),
          (-4.54861962639998, 4.989410770078558),
          (2.193353128786585, 4.144289992394109),
          (0.06994512650339657, 0.4019223150656802),
          (0.002098642248519376, 0.1564790347198236))),
        ('Be',
         ((3.398249705570541, 4.442701786224095),
          (-1.908668860956967, 3.32451542526423),
          (0.03907021175392274, 0.1897728803482149),
          (-0.01116310102107145, 0.08719186146446035),
          (0.009462044653575231, 0.08278090600413406))),
        ('B',
         ((1.472792486393293, 3.749740482818191),
          (-0.4019330421993871, 0.5880665361396737),
          (0.3059989569826894, 0.515639613103011),
          (0.0196144217173168, 0.1213775700806037),
          (0.0009771771060882025, 0.06809824121603139))),
        ('C',
         ((124.4660886213433, 2.421208492560056),
          (-220.3528570789638, 2.305379437524258),
          (195.2353522804791, 2.048519321065642),
          (-98.10793612697996, 1.933525529175474),
          (0.01420230412136232, 0.07689768184783397))),
        ('N',
         ((58.13271507025561, 1.700448564134711),
          (-147.5424090878127, 1.559038526017404),
          (130.1430656496395, 1.415768274731469),
          (-39.61956740841543, 1.278418182054558),
          (0.01059577633314809, 0.05655877984748055))),
        ('O',
         ((29.94740452423624, 1.302839878800107),
          (-77.61012662552783, 1.157941052583095),
          (99.88177646231442, 1.009885493380251),
          (-51.21270055056731, 0.943327971433266),
          (0.00819618954446032, 0.04331976113218256))),
        ('F',
         ((0.9489848945035247, 1.458829331986459),
          (-30.13339230435549, 0.68877999318768),
          (52.79650781273386, 0.6542398693466956),
          (-22.70627037952724, 0.6148361308119943),
          (0.00656997664531441, 0.03428374194950112))),
        ('Ne',
         ((0.5827411922209074, 1.281185731438772),
          (0.3706765618410549, 0.4445208971704776),
          (-0.5467449673508092, 0.198650875510481),
          (0.4140526824802081, 0.1854772466562765),
          (0.005199030808639931, 0.02757383820338858))),
        ('Na',
         ((23.67006039467926, 8.451487735146031),
          (-21.85317861597425, 8.040966004742982),
          (0.5924994481089464, 0.624996000526315),
          (-0.0244652290310244, 0.1324503949472964),
          (0.004839502217065358, 0.02339943620498786))),
        ('Mg',
         ((4.855010476871495, 5.946392738424565),
          (-2.662209064768437, 4.171303125206979),
          (0.478001236085108, 0.3982698081503743),
          (-0.07023070646920648, 0.1618861858374742),
          (0.003989058281040198, 0.01953450563631035))),
        ('Al',
         ((2.834095616075075, 6.662350239805332),
          (-4.28004133378261, 0.5512947222240214),
          (4.421916805483114, 0.5093289634459738),
          (-0.03457744718964006, 0.1117848374253312),
          (0.003523859414060799, 0.01676023518052574))),
        ('Si',
         ((2.871891426116124, 5.084871036429896),
          (-2.061735011951735, 0.4291781853051262),
          (2.171140242044787, 0.3664854341921622),
          (-0.06630736330588019, 0.1197106112969034),
          (0.003010707096705137, 0.01439945361283975))),
        ('P',
         ((2.791518400231514, 3.900659618654661),
          (-4.365068378238221, 0.32982596837715),
          (4.43558455516699, 0.3060899565058886),
          (-0.08096357733994738, 0.1080832325459728),
          (0.002679000179664404, 0.01258944953311868))),
        ('S',
         ((2.679714156101995, 3.068891211999711),
          (-0.4742528222307552, 0.378216702185809),
          (0.5148359489896877, 0.1887218119025488),
          (-0.09583600249907225, 0.09233705900314941),
          (0.002488719638189352, 0.01119208772411441))),
        ('Cl',
         ((2.566248399800203, 2.415949203656124),
          (-0.3388763508285917, 0.421414239310216),
          (1.14584558755515, 0.1095924049758303),
          (-0.9231093165470796, 0.0990955458226753),
          (0.002291680020410422, 0.00999665948927521))),
        ('Ar',
         ((2.459817464140685, 1.940046319888566),
          (-0.3641981770769951, 0.3992410678843988),
          (0.2505844772224745, 0.1174724062744122),
          (-0.05774370295443458, 0.05678037260236218),
          (0.002301438668285837, 0.00915579832920708))),
        ('K',
         ((5.811078786014548, 12.66914833990368),
          (-50.25370965394226, 3.956410396981665),
          (48.86094120598425, 3.683850595771545),
          (0.0740628592048383, 0.1074585175695628),
          (0.0007278027386262219, 0.006655767893915011))),
        ('Ca',
         ((21.17811615241595, 6.396086194317362),
          (-339.0438243174689, 3.740247138917495),
          (322.7569585232967, 3.648884499226053),
          (0.06500776738969967, 0.09450906345146735),
          (0.0006558743665785935, 0.005985206198837586))),
        ('Sc',
         ((12.60351865721486, 6.156256153638529),
          (-276.8753820537026, 3.088735542666793),
          (268.8716039073428, 3.027276632985483),
          (0.0556824178897458, 0.08188747483752157),
          (0.0005770712551453996, 0.005382898320507976))),
        ('Ti',
         ((8.5759577523813, 6.007806688755818),
          (-210.3315634653049, 2.60285856745213),
          (206.0971726015414, 2.553523450511056),
          (0.04777739489772619, 0.07114294840241539),
          (0.0005057164844803206, 0.004856284393837261))),
        ('V',
         ((6.52768433234789, 5.835524793524481),
          (-200.4305768291727, 2.232559523811276),
          (198.015053889994, 2.19786018594229),
          (0.04139115180640056, 0.06239738768285473),
          (0.0004474550243298565, 0.00440383649079958))),
        ('Cr',
         ((3.028317848436913, 8.359115043146318),
          (-95.53939330814323, 1.802637902641032),
          (96.1761562352198, 1.775094889570472),
          (0.03597773159579877, 0.05481444121431137),
          (0.0003914928907184223, 0.003998289689160348))),
        ('Mn',
         ((4.374175506331227, 5.510317055049282),
          (-160.9255109187795, 1.687982164024339),
          (160.2733080603223, 1.666140477777024),
          (0.03123038610491624, 0.0483370390321277),
          (0.000346966021020938, 0.003647469600530682))),
        ('Fe',
         ((3.798100908368596, 5.317126458994331),
          (-91.68935493816872, 1.497130948847481),
          (91.44542521554298, 1.468092418044105),
          (0.02727543440276047, 0.04272478501289549),
          (0.000303379854383771, 0.003327918552318741))),
        ('Co',
         ((3.330378744675443, 5.181359645795432),
          (-77.00175964729671, 1.329151222651394),
          (77.07252217905169, 1.302849289632289),
          (0.02399046690405699, 0.0380645486826371),
          (0.000268256665523942, 0.003050100079915708))),
        ('Ni',
         ((2.969080787252864, 5.0418094910938),
          (-75.74770691290402, 1.182755079216293),
          (76.03982876253073, 1.162165458466299),
          (0.0210162113692967, 0.03374790878830353),
          (0.000231151751135153, 0.002786808620707405))),
        ('Cu',
         ((1.752071452121456, 6.187504979861871),
          (-43.04105234921245, 1.002662636289766),
          (44.07059155435739, 0.9853843113530303),
          (0.01868761540881447, 0.03029847039161176),
          (0.0002017273247916241, 0.002558555987488791))),
        ('Zn',
         ((2.466371104994591, 4.910280784938159),
          (-61.46785413325375, 0.9678985203229922),
          (62.01769452374813, 0.9512838347753555),
          (0.01641601739314162, 0.02696009676675663),
          (0.0001724871175953809, 0.002341096110462537))),
        ('Ga',
         ((2.760102031084283, 6.101282245376627),
          (-34.44526142074679, 0.7651433135534649),
          (35.22622672440163, 0.7513286233828597),
          (0.01320671969994199, 0.02248796343172512),
          (0.0001259455609282457, 0.002067373742787123))),
        ('Ge',
         ((3.1824163526, 5.017190408609147),
          (-52.45140378111667, 0.7123957644377982),
          (52.96908271622185, 0.702280192528194),
          (0.01140961685918648, 0.01967472956940154),
          (9.509543581450579e-05, 0.001841466144940115))),
        ('As',
         ((3.45642969119604, 4.013580160329452),
          (-33.31760444317212, 0.6623557780506291),
          (33.57121938553322, 0.6457719410560738),
          (0.00979002295640342, 0.01709193532310115),
          (6.534348622651725e-05, 0.001603016028394201))),
        ('Se',
         ((3.649050478019264, 3.250432671125934),
          (-43.68516622212386, 0.609666201650289),
          (43.69202886011548, 0.5969713008021232),
          (0.008449022841991444, 0.01485545127403623),
          (3.786114741100382e-05, 0.00133625587356563))),
        ('Br',
         ((3.838463122242895, 2.611894705732323),
          (-52.27234710112339, 0.5661950628747182),
          (51.98612794956596, 0.5552793266998739),
          (0.007339559893380448, 0.01297464694324407),
          (1.646942079513399e-05, 0.001029865368558757))),
        ('Kr',
         ((4.025410303108274, 2.1364838144374),
          (-46.30423320789298, 0.5265911664541312),
          (45.72136819041012, 0.5141367844645776),
          (0.006353379596253157, 0.01128072422420068),
          (1.33477845255743e-06, 0.0004880898579409704))),
        ('Rb',
         ((3.38975351595394, 20.57448143681711),
          (2.143483486791724, 1.910799452185164),
          (0.3543226035109782, 0.1974105893966039),
          (0.003740093400856642, 0.00813459465343989),
          (3.003425023422297e-07, 0.0002926857910569511))),
        ('Sr',
         ((4.77092509299826, 13.36688813304128),
          (1.475978501552834, 1.337383795771967),
          (0.3044513555441887, 0.1775323694377828),
          (0.003594749819167283, 0.007791050330018268),
          (3.00085549228279e-07, 0.0002822551396485372))),
        ('Y',
         ((4.60721019875234, 10.86869055571519),
          (1.428018510398403, 1.311374558731241),
          (0.2955810455777538, 0.1680228707847112),
          (0.003389978408528951, 0.007359645453504783),
          (2.668629749690224e-07, 0.0002623432810213873))),
        ('Zr',
         ((4.311754534068716, 9.458965805174902),
          (1.493315780393395, 1.330636228452456),
          (0.2812360501288841, 0.1565068714444536),
          (0.003093377384557471, 0.006824105238117352),
          (2.580244485125306e-07, 0.0002498188791107023))),
        ('Nb',
         ((3.111790391134954, 10.69031413430237),
          (2.202590609458031, 1.653163561589331),
          (0.2703307749100209, 0.1451151857189699),
          (0.002687991947510981, 0.006139563515828505),
          (2.325494833477792e-07, 0.0002322402359376725))),
        ('Mo',
         ((2.831059684320536, 10.43571957589507),
          (2.348581374896745, 1.604828686745972),
          (0.2451058884296964, 0.1316969347746069),
          (0.002352821082185159, 0.005549779013833459),
          (2.312708383438592e-07, 0.0002227474637648977))),
        ('Tc',
         ((2.571798593233526, 10.16431171313775),
          (2.45633741957203, 1.534419192445505),
          (0.2206584408813715, 0.1191386197541104),
          (0.002055314180124382, 0.005018532408829888),
          (2.321329477936922e-07, 0.0002145903791208612))),
        ('Ru',
         ((2.332300303539466, 9.921674601599594),
          (2.535780254890496, 1.455856688087881),
          (0.198208042136026, 0.1076821878733493),
          (0.001761201304600535, 0.004472435456549673),
          (1.981294115460245e-07, 0.0001965747162324994))),
        ('Rh',
         ((2.1135253485947, 9.659137258597623),
          (2.586363161550138, 1.371066569343296),
          (0.1770639138636865, 0.09703530275846488),
          (0.001497370510030569, 0.003971285090887921),
          (2.054814455556224e-07, 0.0001913551907377983))),
        ('Pd',
         ((0.6421597961826873, 5.974797502634061),
          (2.979148144263289, 1.433594325412777),
          (0.1681544260042708, 0.0909868401172677),
          (0.001337442138784078, 0.003624101371161622),
          (1.914109682468616e-07, 0.000180688914416045))),
        ('Ag',
         ((1.553172166803896, 8.156202357589565),
          (2.639303646999888, 1.216008874818015),
          (0.1420154869567882, 0.07900988649158734),
          (0.001008504600997609, 0.00296547901363949),
          (1.946384299730244e-07, 0.0001745930959191462))),
        ('Cd',
         ((61.53078519928602, 3.114681025332473),
          (-78.60167412015821, 2.760169833885745),
          (21.55012926027705, 1.935513123247224),
          (0.1376850156641916, 0.07224683472602932),
          (0.000324644930946521, 0.001170016296246845))),
        ('In',
         ((4.222321779015246, 6.072655104032275),
          (-26.41213183532026, 1.645501789593879),
          (27.28528527087469, 1.522570749395068),
          (0.121617901884138, 0.06565079146952256),
          (0.0003068835463715258, 0.001120938514730356))),
        ('Sn',
         ((5.142220746420537, 5.272726364744845),
          (-25.49454137625767, 1.531949592091483),
          (25.74144875486019, 1.402575250400872),
          (0.1117782275921996, 0.06116853872630869),
          (0.0002936473847377427, 0.001075608153945223))),
        ('Sb',
         ((6.241640318318837, 4.269841080826875),
          (-93.38687244195522, 1.394077461402502),
          (92.63328758298843, 1.355668549104345),
          (0.1034126922212987, 0.05726679496521998),
          (0.0002818484268943907, 0.001033079542820674))),
        ('Te',
         ((7.37743301813403, 3.469177577838978),
          (-126.0251069316281, 1.297598108867366),
          (124.1284050040956, 1.26771067646066),
          (0.09599783718185698, 0.05377717501794231),
          (0.000271072216398882, 0.0009930845770291768))),
        ('I',
         ((9.644006662721232, 2.726455453665441),
          (-122.9244353501125, 1.23723425850866),
          (118.6825648165673, 1.200620368722498),
          (0.0895025947025539, 0.05066786822851999),
          (0.000261276121490475, 0.0009554378833834973))),
        ('Xe',
         ((15.54517496748606, 2.106373408654927),
          (-118.2410278567446, 1.208603761295122),
          (108.009524963197, 1.15395270567214),
          (0.08362593419922217, 0.04781893912748243),
          (0.0002519918624290739, 0.0009198862575926131))),
        ('Cs',
         ((4.287087391816923, 22.65878707985415),
          (3.23250665422965, 2.237973864700642),
          (0.6740295335617063, 0.3689955686643163),
          (0.06189080833876976, 0.04022665752984844),
          (0.00023561205295219, 0.0008837618908780351))),
        ('Ba',
         ((6.244751873904615, 15.14313541909856),
          (2.351722714163885, 1.45379000604814),
          (0.474279373222252, 0.3208356463878018),
          (0.06381138742868499, 0.04043545320946998),
          (0.0002346512805627924, 0.0008540811312800327))),
        ('La',
         ((6.097881795995097, 12.42885443158548),
          (2.19495164736675, 1.505359923520235),
          (0.5481727919600226, 0.3337380397171243),
          (0.06166695732226626, 0.03877445534999988),
          (0.0002268073558647142, 0.000824049884064867))),
        ('Ce',
         ((5.795268796472405, 14.2801055058256),
          (2.370226641078433, 1.359690157191345),
          (0.4713987569011149, 0.3020173496635141),
          (0.05743682605878668, 0.03664367981470077),
          (0.0002189794892596593, 0.0007954345265183711))),
        ('Pr',
         ((5.604062553775258, 13.95174902305747),
          (2.357962595618129, 1.312397549174392),
          (0.4760010985728824, 0.2949337011929565),
          (0.05441233743152471, 0.03486015424623241),
          (0.0002114146022051498, 0.0007682616826618097))),
        ('Nd',
         ((5.429083919697703, 13.650364942766),
          (2.336873253608803, 1.267598413903199),
          (0.4833735541048819, 0.2886106815769299),
          (0.0514654964557479, 0.03312918074466093),
          (0.0002037761328637611, 0.0007423484838012663))),
        ('Pm',
         ((5.267744450894446, 13.36020968273946),
          (2.30855813326305, 1.225858565869417),
          (0.4932654790260128, 0.2829196321278669),
          (0.04863562797416966, 0.03147353971271379),
          (0.0001963088423206752, 0.0007176580250696153))),
        ('Sm',
         ((5.126804285170776, 13.1581501026129),
          (2.269255340083806, 1.181295082433371),
          (0.5042093004533025, 0.2771070382190629),
          (0.0456923992416222, 0.02979576045495446),
          (0.0001886750504938862, 0.0006940110991990983))),
        ('Eu',
         ((4.979623597498092, 12.83926630780338),
          (2.241830874556695, 1.14705446420486),
          (0.5129339614028937, 0.2703871609352437),
          (0.04298018484983344, 0.0282418714952602),
          (0.0001813816924857876, 0.0006714866031739924))),
        ('Gd',
         ((5.078358300456114, 10.51327254690473),
          (1.95744027181659, 1.117649412815246),
          (0.5928259832213751, 0.2843867418336752),
          (0.04195020341439256, 0.02726633276413471),
          (0.0001752410915283168, 0.0006503108607073037))),
        ('Tb',
         ((4.711616366573001, 12.31094159485391),
          (2.172619507865789, 1.087437967674925),
          (0.5397176013428657, 0.2598049655092755),
          (0.03797960045478111, 0.02532899238951897),
          (0.0001669237635648417, 0.0006292785669619112))),
        ('Dy',
         ((4.590755044851466, 12.06567406233699),
          (2.135723730365674, 1.058117371158644),
          (0.5513555600941525, 0.2539944389186618),
          (0.03560584067185138, 0.02395087396496492),
          (0.0001598240168567898, 0.0006094827484332881))),
        ('Ho',
         ((4.484001106267108, 11.87492506915706),
          (2.089043470785398, 1.028284937087896),
          (0.5659326183161046, 0.248907807908076),
          (0.0332703590211063, 0.02258440655321478),
          (0.0001524456102828921, 0.0005903744451594245))),
        ('Er',
         ((4.376651407869632, 11.66530807171395),
          (2.046451503836341, 1.003147696754953),
          (0.5806628605915877, 0.2441532926863124),
          (0.03123885283277667, 0.0213618580000731),
          (0.0001453748696625357, 0.0005721103384547666))),
        ('Tm',
         ((4.283083182474195, 11.49619059566834),
          (1.995380014537307, 0.9770391020188206),
          (0.5970535713325043, 0.2394291329758291),
          (0.02909516688695599, 0.02009122333272558),
          (0.0001380647690375216, 0.0005543771384922663))),
        ('Yb',
         ((4.195638407371233, 11.41077504566523),
          (1.943332859786456, 0.9490119244548774),
          (0.6124466172854658, 0.2349503265353047),
          (0.02715152076949324, 0.01890515762809755),
          (0.0001305947873519297, 0.0005372336492097111))),
        ('Lu',
         ((4.356925932963843, 9.29434514718569),
          (1.695892047773159, 0.91050004595742),
          (0.6639045200684704, 0.2387465959113764),
          (0.02630200187602814, 0.01820985425464623),
          (0.0001254973184998238, 0.0005215593719635622))),
        ('Hf',
         ((4.331384056649235, 7.876844338102884),
          (1.527648647286807, 0.9425156426277488),
          (0.735795922891238, 0.241699480039806),
          (0.02495263232014026, 0.0172899894448509),
          (0.0001187408525810578, 0.0005058346313135256))),
        ('Ta',
         ((4.19726001319642, 6.936740248944572),
          (1.468548067747088, 1.017252766183483),
          (0.7839312482110882, 0.2389489397150706),
          (0.02324940948643951, 0.0162332433075381),
          (0.000111261358607286, 0.000490239004021149))),
        ('W',
         ((3.976297158190774, 6.296857792287749),
          (1.522926842573418, 1.112899511576911),
          (0.7977062463905613, 0.2310570406784734),
          (0.02136667415586852, 0.01510135455672045),
          (0.0001030786893857173, 0.0004746824771050258))),
        ('Re',
         ((3.751443814398119, 5.797546360829538),
          (1.627688029776327, 1.182236310777107),
          (0.7817567554542718, 0.2199135860247702),
          (0.01951708039122916, 0.01398104554814566),
          (9.431998042934275e-05, 0.0004591271258298386))),
        ('Os',
         ((3.484015173387116, 5.439988460809537),
          (1.793779204169655, 1.227921341401281),
          (0.7448783566919203, 0.2062088569929092),
          (0.01754277851624345, 0.0127899401721146),
          (8.4487235156886e-05, 0.0004430322267874786))),
        ('Ir',
         ((1.599565781988442, 5.792444473855675),
          (2.975344521941205, 1.55300982973226),
          (0.6950926783668822, 0.1886263359366482),
          (0.0148779627458688, 0.01117634706624533),
          (6.905495791015833e-05, 0.0004227723616555514))),
        ('Pt',
         ((2.040215639757283, 6.658194296096275),
          (2.899226346248255, 1.413379237789324),
          (0.6363440838157977, 0.174000104502179),
          (0.01320719195954083, 0.01006878045302105),
          (5.673821925001312e-05, 0.0004030771056922231))),
        ('Au',
         ((1.675934670648708, 5.522310932114025),
          (3.004866029697293, 1.380072230071963),
          (0.5953400131616355, 0.1622292376559454),
          (0.01171631866230948, 0.009018148904165756),
          (4.296782976398171e-05, 0.0003792776674776671))),
        ('Hg',
         ((2.235228504431053, 5.020309889602403),
          (2.682766386519949, 1.230775905837777),
          (0.5551949262124333, 0.1522481229928636),
          (0.01072733543662587, 0.008283991169062104),
          (3.284739920462629e-05, 0.0003562419389317589))),
        ('Tl',
         ((2.8034273741251, 6.558768728045342),
          (2.7188278796602, 1.169724225166679),
          (0.5224759153919996, 0.1435566918001781),
          (0.00984537836971042, 0.007619765262300398),
          (2.345245265083162e-05, 0.0003296276739029854))),
        ('Pb',
         ((3.608610209977714, 6.581625946219628),
          (2.450567747371983, 1.027728526605885),
          (0.478639500107257, 0.1335336806347209),
          (0.008872142351692152, 0.006848612389484298),
          (1.040019329094673e-05, 0.000276388875456666))),
        ('Bi',
         ((4.242099010573159, 5.7528011553608),
          (2.099943420558279, 0.8739014891954615),
          (0.4328366313799229, 0.1235999561805247),
          (0.008020215703810592, 0.006176003330585702),
          (7.217850308513404e-07, 0.0001414951776172309))),
        ('Po',
         ((4.636200956689623, 4.888252997809829),
          (1.780633114269871, 0.7563105268800747),
          (0.4037304439527385, 0.1173023644522492),
          (0.00768539554668674, 0.005930343942428942),
          (8.954159028076436e-08, 7.666686638730287e-05))),
        ('At',
         ((4.965922505950705, 4.091293787874963),
          (1.438155615070337, 0.6292289966025513),
          (0.3712992005792981, 0.1110966786955352),
          (0.007472564284502011, 0.005772350103477618),
          (1.141152841525851e-07, 8.085118938793825e-05))),
        ('Rn',
         ((5.306156144750332, 3.488007354816557),
          (1.117331602360721, 0.4811907411497712),
          (0.3158587231108543, 0.1018744679457531),
          (0.0072034224231021, 0.005592453744170787),
          (1.073553305458635e-07, 7.795066735407367e-05))),
        ('Fr',
         ((4.52053399042336, 19.44822342329488),
          (4.106953979091246, 1.898246731559969),
          (0.7139468785037285, 0.1695535635953418),
          (0.01692940276874539, 0.01148195675489342),
          (8.574921291917244e-05, 0.0003461220382579827))),
        ('Ra',
         ((6.524010720018733, 14.00925542989749),
          (3.207870807456613, 1.326350359616653),
          (0.5404787743496376, 0.1314085683860361),
          (0.008782788068988532, 0.006286474345204096),
          (6.910106029491315e-06, 0.000227839981458951))),
        ('Ac',
         ((6.896028535596191, 11.07638256703987),
          (2.835141545365232, 1.17132616290503),
          (0.5035068818888151, 0.1234946513917999),
          (0.008022945109667069, 0.005735155957904971),
          (9.204009547587397e-08, 7.197075418025232e-05))),
        ('Th',
         ((7.093749001626302, 9.094737951659447),
          (2.52912373903194, 1.063916670331612),
          (0.482107819888889, 0.1186194466218779),
          (0.00771936674145111, 0.005539457088950347),
          (7.271141994250414e-08, 6.584947305284627e-05))),
        ('Pa',
         ((6.434013247972842, 10.25968513072979),
          (2.970999705357888, 1.131774523061633),
          (0.460796651787848, 0.1127759353623823),
          (0.007240331257754553, 0.005274595833531321),
          (6.362366643091279e-08, 6.192638240887597e-05))),
        ('U',
         ((6.210708267840199, 10.02141058591628),
          (3.039344538256235, 1.103998801558834),
          (0.4373399844391628, 0.1072189731624539),
          (0.006807147641473166, 0.005024321308506824),
          (6.182293277428513e-08, 6.005069941971958e-05))),
        ('Np',
         ((6.004315983089865, 9.817930467191061),
          (3.09431654531418, 1.069787050680294),
          (0.412808487716417, 0.1016352914486458),
          (0.006408916578800669, 0.00479524846888725),
          (6.730073762160754e-08, 6.028065987264629e-05))),
        ('Pu',
         ((5.200617168103042, 11.20383101944067),
          (3.49849440367144, 1.128463695469282),
          (0.4054149311313203, 0.09893334726861316),
          (0.006223437008265299, 0.004659174319779507),
          (6.008593295434409e-08, 5.725906389623341e-05))),
        ('Am',
         ((5.025338600360291, 10.97721906976287),
          (3.518439882851541, 1.084772148326308),
          (0.3819503494462722, 0.09367468748538359),
          (0.00582110208096211, 0.004426823468734856),
          (6.526093388297119e-08, 5.738568370505115e-05))),
        ('Cm',
         ((5.346561002606197, 9.231183797524494),
          (3.224684665609108, 0.9728362291938352),
          (0.3494617526254452, 0.08705962209665623),
          (0.005292517567488151, 0.004130119644650323),
          (6.159176302362374e-08, 5.494638944496757e-05))),
        ('Bk',
         ((5.225823503340047, 9.071357066187183),
          (3.228188739952953, 0.9311242774566348),
          (0.3270988788238514, 0.08207206020598455),
          (0.00488882575592239, 0.003890296559934166),
          (5.212722694193633e-08, 5.103679766366416e-05))),
        ('Cf',
         ((4.586412477535479, 10.31861020923352),
          (3.518695956651012, 0.9572788378292328),
          (0.3192617142012054, 0.07968074314438807),
          (0.004729796479856258, 0.003771931999968489),
          (5.513244808062041e-08, 5.097508603760111e-05))),
        ('Es',
         ((4.457994806754126, 10.08906933834013),
          (3.508672126376623, 0.9194464473616226),
          (0.3013753805283154, 0.07564191777863292),
          (0.004407637462144819, 0.003570264651248093),
          (4.887879032285916e-08, 4.804951743958113e-05))),
        ('Fm',
         ((4.338975764011709, 9.963309372238362),
          (3.491850988964034, 0.8780839569829744),
          (0.2814710990162868, 0.07116371157687776),
          (0.004002092714990461, 0.003321524042150777),
          (5.52929808195514e-08, 4.851031794603728e-05))),
        ('Md',
         ((4.227294704031012, 9.724006020400314),
          (3.472492275106616, 0.8428737759602104),
          (0.2648222294968986, 0.06735347439748511),
          (0.003690728778331929, 0.003123646062633208),
          (6.258714262002473e-08, 4.912170970176192e-05))),
        ('No',
         ((4.109517024430204, 9.677359945101202),
          (3.457991325227507, 0.8069400424708172),
          (0.2470873512223867, 0.0632815436954167),
          (0.003304239209409952, 0.002875446396412092),
          (5.991049262319316e-08, 4.706791536975981e-05))),
        ('Lr',
         ((4.521474211983788, 8.28309906861142),
          (3.202129855878044, 0.7319189581253939),
          (0.2230287269564517, 0.0580942773018655),
          (0.002817164538920297, 0.002561680160474449),
          (4.064279540386205e-08, 4.038165155290065e-05))))
)
