Număr PDF-uri testate: 40
Număr total pagini: 2248
Prompturi:

I1 = întrebare generală / rezumat
I2 = întrebare multimodală
I3 = quiz

# **Qwen**

I1:  
	medie = 126.78s, 
	deviație standard = 29.05s, 
	minim = 58.63s, 
	maxim = 199.47s

I2: 
	medie = 67.01s, 
	deviație standard = 18.45s, 
	minim = 24.38s, 
	maxim = 103.90s

I3: 
	medie = 52.44s, 
	deviație standard = 10.96s, 
	minim = 32.67s, 
	maxim = 76.80s

Total model:
	medie generală = 82.08s
	deviație standard = 38.35s
	minim = 24.38s
	maxim = 199.47s

# **Gemma**

I1: 
	medie = 138.58s, 
	deviație standard = 26.27s, 
	minim = 73.53s, 
	maxim = 191.

I2:
	medie = 54.67s, 
	deviație standard = 13.97s, 
	minim = 29.83s, 
	maxim = 84.60s

I3:
	medie = 70.47s, 
	deviație standard = 11.43s, 
	minim = 46.59s, 
	maxim = 96.20s

Total model:
	medie generală = 87.91s
	deviație standard = 40.86s
	minim = 29.83s
	maxim = 191.30s

# Deviația standard 

	s = sqrt( Σ(xi - media)^2 / (n - 1) )
	
		xi = fiecare timp măsurat
		media = media timpilor
		n = numărul de valori
		s = deviația standard


Deviație standard mică = timpi apropiați între ei
Deviație standard mare = timpi variați mult între documente

# **Comparație Directă**

I1: Qwen = 126.78s, Gemma = 138.58s
	Qwen a fost mai rapid cu 11.79s în medie.

I2: Qwen = 67.01s, Gemma = 54.67s
	Gemma a fost mai rapid cu 12.34s în medie.

I3: Qwen = 52.44s, Gemma = 70.47s
	Qwen a fost mai rapid cu 18.03s în medie.

Global:
	Qwen = 82.08s
	Gemma = 87.91s
	Qwen a fost mai rapid global cu 5.83s în medie.

# Medii

Media globală pe pagină = total timp / total pagini
Media rapoartelor pe document = media valorilor timp document / pagini document

Qwen:
	I1 = 2.26s/pagină global
	I2 = 1.19s/pagină global
	I3 = 0.93s/pagină global
	Total toate prompturile = 1.46s/pagină/prompt

Gemma:
	I1 = 2.47s/pagină global
	I2 = 0.97s/pagină global
	I3 = 1.25s/pagină global
	Total toate prompturile = 1.56s/pagină/prompt


## Corelația dintre numărul de pagini și timp este moderat-pozitivă (coeficientul de corelație - r)

	Qwen: r între 0.61 și 0.68
	Gemma: r între 0.65 și 0.71

	r aproape de 1 = când numărul de pagini crește, timpul tinde să crească
	r aproape de 0 = nu există o relație liniară clară
	r aproape de -1 = când numărul de pagini crește, timpul tinde să scadă


