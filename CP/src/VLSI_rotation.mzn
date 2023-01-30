include "globals.mzn";
include "cumulative.mzn";
include "lex_lesseq.mzn";
include "diffn.mzn";

% VARIABLES
int: w; % width of the plate
int: n; % number of rectangles
bool: first_rectangle_rot;
set of int: RECTANGLES = 1..n;
array[RECTANGLES] of int: x_components;
array[RECTANGLES] of int: y_components;
int: max_h = sum(y_components); % the maximum height is the sum of all the y components
int: min_h; % the minimum height is AREA / width of the plate
var min_h..max_h: h; % h have to variate between h_min and h_max
int: max_coord_x = w - min(x_components); % is useless to assign all the possibile coordinates from 0 to w; indeed, we can arrive at most to w - the minimum x components
int: max_coord_y = max_h - min(y_components);
array[RECTANGLES] of var 0..max_coord_x: x_positions; % array for the x coordinates (i.e. in the first element is the first rectangle with 0-w possible values)
array[RECTANGLES] of var 0..max_coord_y: y_positions;

% ROTATION PART
array[RECTANGLES] of var 0..max(max_h, w): x_components_rot; % store the x component of the rectangle i. The coordinates have to the max between w and h_max
array[RECTANGLES] of var 0..max(max_h, w): y_components_rot;
array[RECTANGLES] of var bool: rotations; % [false false true ...] --> a list of BOOL that indicates: true the rectangles is rotated, false otherwise
constraint forall (i in RECTANGLES) (if rotations[i] then y_components_rot[i] == x_components[i] /\ x_components_rot[i] == y_components[i] else x_components_rot[i] == x_components[i] /\ y_components_rot[i] == y_components[i] endif);

% CONSTRAINTS
constraint cumulative(y_positions, y_components_rot, x_components_rot, w); % cumulative
constraint cumulative(x_positions, x_components_rot, y_components_rot, h); % cumulative
constraint diffn(x_positions, y_positions, x_components_rot, y_components_rot); % no overlapping
constraint forall(i in 1..n)(y_components_rot[i] + y_positions[i] <= h); % y component and y coordinate for rectangle i must be <= h
constraint forall(i in 1..n)(x_components_rot[i] + x_positions[i] <= w);



solve :: seq_search([
    int_search([h], smallest, indomain_min),
    restart_constant(5000)])
    minimize h;
