%opengl software
curve_unsup = load('./curve_geo_error_unsup_non_iso.mat');
curve_sup = load('./curve_geo_error_sup_non_iso.mat');
curve_pnpp = load('./curve_geo_error_rand_pnpp.mat');
curve_pn = load('./curve_geo_error_sup_paper_high_dense.mat');
mean_unsup = mean(curve_unsup.mean_curves(2,:))
mean_sup = mean(curve_sup.mean_curves(2,:))
mean_pn = mean(curve_pn.mean_curves(2,:))
mean_pnpp = mean(curve_pnpp.mean_curves(2,:))
%plot(curve_unsup.thr, curve_unsup.mean_curves(1,:), 'LineWidth',2);
%hold on
plot(curve_unsup.thr, curve_unsup.mean_curves(2,:), 'LineWidth',2);
hold on
%plot(curve_sup.thr, curve_sup.mean_curves(1,:), 'LineWidth',2);
plot(curve_sup.thr, curve_sup.mean_curves(2,:), 'LineWidth',2);
%plot(curve_sup.thr, curve_hk.mean_curves(1,:), 'LineWidth',2);
plot(curve_sup.thr, curve_pnpp.mean_curves(2,:), 'LineWidth',2);
%plot(curve_sup.thr, curve_tnet.mean_curves(2,:), 'LineWidth',2);
grid on
%plot(curve_sup.thr, repmat(mean_sup, 1, size(curve_sup.thr, 2)));
%plot(curve_unsup.thr, repmat(mean_unsup, 1, size(curve_unsup.thr, 2)));
legend(['unsupervised'],['supervised'],['PointNet++'])
hold off

%plot(curve_unsup.thr, curve_unsup.mean_curves(1,:), 'LineWidth',2);
%hold on
plot(curve_unsup.thr, curve_pn.mean_curves(2,:), 'LineWidth',2);
hold on
%plot(curve_sup.thr, curve_sup.mean_curves(1,:), 'LineWidth',2);
%plot(curve_sup.thr, curve_sup.mean_curves(2,:), 'LineWidth',2);
%plot(curve_sup.thr, curve_hk.mean_curves(1,:), 'LineWidth',2);
plot(curve_sup.thr, curve_pnpp.mean_curves(2,:), 'LineWidth',2);
%plot(curve_sup.thr, curve_tnet.mean_curves(2,:), 'LineWidth',2);
grid on
%plot(curve_sup.thr, repmat(mean_sup, 1, size(curve_sup.thr, 2)));
%plot(curve_unsup.thr, repmat(mean_unsup, 1, size(curve_unsup.thr, 2)));
legend(['PointNet'],['PointNet++'])
hold off

%hold on
for i=1:2
    trapz(curve_unsup.thr, curve_unsup.mean_curves(i,:))
    trapz(curve_unsup.thr, curve_sup.mean_curves(i,:))
end
