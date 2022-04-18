clear all

load('./curve_geo_error_rand_pnpp.mat');
%load('dd2_pnpp.mat');

faces = faces + 1;
area = [];

for i=1:size(match, 1)
    x  = calc_err_curve(geo_err(i,:)', thr);
    area = [area trapz(thr, x)];
    %plot(thr, x, 'LineWidth', 2), ylim([0, 100]), pause;
end

mean(area)
ans = 96.0652
[a, b] = sort(area,'ascend')

for i=b
    number_1 = i;
    display(area(number_1));
    num_1 = source(1,number_1) + 1;
    num_2 = target(1,number_1) + 1;    
    number_2 = find(source == target(1,number_1) & target == source(1,number_1));
    shape_1.VERT = squeeze(vertices(num_1,:,:));
    shape_2.VERT = squeeze(vertices(num_2,:,:));
    shape_1.TRIV = faces;
    shape_2.TRIV = faces;
    shape_1.n = size(shape_1.VERT, 1);
    shape_2.n = size(shape_2.VERT, 1);
    %match_no = match(number_1,:);
    match_this_sup_1 = match(number_1,:);
    match_this_sup_2 = match(number_2,:);
    %match_this_unsup_1 = unsup_a.match(number_1,:);
    %match_this_unsup_2 = unsup_a.match(number_2,:);
    %dist = pdist2(squeeze(desc(num_1,:,:)), squeeze(desc(num_2,:,:)));
    %match = matchpairs(dist, 1000, 'min');
    %match = match(:,1);
    %subplot(211), trisurf(shape_1.TRIV, shape_1.VERT(:,1), shape_1.VERT(:,2), shape_1.VERT(:,3), 1:1000, 'EdgeColor', 'None'); axis equal;
    %subplot(212), trisurf(shape_2.TRIV, shape_2.VERT(:,1), shape_2.VERT(:,2), shape_2.VERT(:,3), match,'EdgeColor', 'None'); axis equal;
    cmap_sup = plot_3Dcorrespondence_mod(shape_1, shape_2, [[1:size(match, 2)]' , (match_this_sup_1 + 1)'], [[1:size(match, 2)]' , (match_this_sup_2 + 1)']); pause
    %cmap_unsup = plot_3Dcorrespondence_mod(shape_1, shape_2, [[1:size(sup_a.match, 2)]' , (match_this_unsup_1 + 1)'], [[1:size(sup_a.match, 2)]' , (match_this_unsup_2 + 1)']); pause
    %plot_3Dcorrespondence(shape_1, shape_2, [[1:size(match_no, 2)]', (match_no+1)']);pause;
    %pause 
end

%plot(thr, calc_err_curvce(geo_err(num,:)',thr), 'LineWidth',2);
%grid on
