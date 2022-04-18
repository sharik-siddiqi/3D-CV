%opengl software

curve_unsup = load('./curve_geo_error_unsup_non_iso.mat'); %Performance on non-isometric shapes (Unsupervised)
curve_sup = load('./curve_geo_error_sup_non_iso.mat'); %Performance on non-isometric shapes (Supervised)

% Mean error calculation per vertex on Test dataset
mean_unsup = mean(curve_unsup.mean_curves(2,:))
mean_sup = mean(curve_sup.mean_curves(2,:))

%Results for supervised and unsupervised cases on test dataset in terms of geodesic error curves
plot(curve_unsup.thr, curve_unsup.mean_curves(2,:), 'LineWidth',2);
hold on
plot(curve_sup.thr, curve_sup.mean_curves(2,:), 'LineWidth',2);
grid on
legend(['unsupervised'],['supervised'])
hold off

for i=1:2
    trapz(curve_unsup.thr, curve_unsup.mean_curves(i,:))
    trapz(curve_unsup.thr, curve_sup.mean_curves(i,:))
end
