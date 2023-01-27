include "globals.mzn";
include "cumulative.mzn";
include "diffn.mzn";
include "lex_lesseq.mzn";

% VARIABLES
int: w; % width of the plate
int: n; % number of rectangles
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

% CONSTRAINTS
constraint cumulative(y_positions, y_components, x_components, w); % cumulative
constraint cumulative(x_positions, x_components, y_components, h); % cumulative
constraint diffn(x_positions, y_positions, x_components, y_components); % no overlapping
constraint forall(i in 1..n)(y_components[i] + y_positions[i] <= h); % y component and y coordinate for rectangle i must be <= h
constraint forall(i in 1..n)(x_components[i] + x_positions[i] <= w);
constraint y_positions[1] = 0; % the first rectangle is set in (0,0) coordinates (NOTE: the first rectangle is the one with the highest y component)
constraint x_positions[1] = 0;
constraint lex_lesseq([x_positions[i] | i in 1..n], [max_coord_x - x_positions[i] - x_components[i] | i in 1..n]); % symmetry breaking constraint


solve :: seq_search([
    int_search([h], smallest, indomain_min),
    restart_constant(5000)])
    minimize h;
