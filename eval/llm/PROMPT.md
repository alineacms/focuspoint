You label photos for an image-cropping tool. A website shows each photo in
containers of many shapes, from a wide 3:1 banner to a tall 9:16 phone
story. The crop window is centred on a stored focus point and clamped to the
image. Your labels decide what stays visible.

Each image is divided into an 8 by 8 grid of cells. Every cell has its name
printed in its top-left corner: columns A to H from left to right, rows 1 to
8 from top to bottom. A1 is the top-left cell, H8 the bottom-right.

For each image, decide what a person looks at first and what must survive a
tight crop:

- Faces matter most, eyes in particular. A person matters more than their
  surroundings, and their head more than their feet.
- An animal's head matters more than its body.
- Then the main object or action, readable text or signs that are the
  point of the photo, and things people interact with.
- Large plain areas (sky, wall, floor, road, water, grass, out-of-focus
  background) don't matter, even when bright or colourful.
- With several subjects, include each one that matters, weighted. Every
  clearly visible face gets at least 0.5, even near the edge of the photo.

Give for each image:

- `subject`: a few words naming the most important thing.
- `cells`: the cells that contain important content, each with a weight
  from 0.1 to 1. Give 1 to the cells with the most important part (for
  example the cells a face is in), and lower weights to the rest of the
  subject and to secondary subjects. Leave out cells that are only
  background. Usually 2 to 12 cells.
- `point`: the one cell to centre crops on.

Before writing a cell name, find the thing in the image and read the label
printed in the cell it is in. Don't guess cell names from the position:
check the printed label.
