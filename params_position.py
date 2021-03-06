#Zoom, pan, crop parameters used by 6 sets of examples in ”input”

#1
example1 = dict(number = 1,
fore_path = 'input/fore/1.png',
mask_path = 'input/mask/1.png',
back_path = 'input/back/1.jpg',
# Zoom
zoomSize = 1,
# Pan
Vertical = -100,
Horizontal = -50,
# Crop
Left = 0,
Right = -1,
Top = 50,
Bottom = 600+1,)


#2
example2 = dict(number = 2,
fore_path = 'input/fore/2.jpg',
mask_path = 'input/mask/2.png',
back_path = 'input/back/2.jpg',
# Zoom
zoomSize = 1.5,
# Pan
Vertical = 100,
Horizontal = 0,
# Crop
Left = 550,
Right = 1150+1,
Top = 200,
Bottom = 800+1)


#3
example3 = dict(number = 3,
fore_path = 'input/fore/3.jpg',
mask_path = 'input/mask/3.png',
back_path = 'input/back/3.jpg',
# Zoom
zoomSize = 1,
# Pan
Vertical = -5,
Horizontal = 0,
# Crop
Left = 0,
Right = -1,
Top = 0,
Bottom = -1)


#4
example4 = dict(number = 4,
fore_path = 'input/fore/4.jpg',
mask_path = 'input/mask/4.png',
back_path = 'input/back/4.jpg',
# Zoom
zoomSize = 0.5,
# Pan
Vertical = 0,
Horizontal = 50,
# Crop
Left = 0,
Right = 800+1,
Top = 0,
Bottom = 620+1)


#5
example5 = dict(number = 5,
fore_path = 'input/fore/5.jpg',
mask_path = 'input/mask/5.png',
back_path = 'input/back/5.jpg',
# Zoom
zoomSize = 0.3,
# Pan
Vertical = 180,
Horizontal = 0,
# Crop
Left = 0,
Right = -1,
Top = 300,
Bottom = -1)


#6
example6 = dict(number = 6,
fore_path = 'input/fore/6.png',
mask_path = 'input/mask/6.png',
back_path = 'input/back/6.jpg',
# Zoom
zoomSize = 0.7,
# Pan
Vertical = 50 ,
Horizontal = 150,
# Crop
Left = 200,
Right = -1,
Top = 150,
Bottom = 700+1)

example_all = [example1, example2, example3, example4, example5, example6]
