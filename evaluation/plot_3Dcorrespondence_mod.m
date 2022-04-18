function [cmapN] = plot_3Dcorrespondence_mod(M, N, corr, corr_2)
    figure,
    cmapM = create_colormap(M, M);
    subplot(121), trisurf(M.TRIV, M.VERT(:,1), M.VERT(:,2), M.VERT(:,3), 1:M.n, 'EdgeAlpha', 0), axis equal, axis off;
    colormap(gca, cmapM);
    %freezeColors();
    cmapN = zeros(N.n, 3);
    cmapN(corr(:,2), :) = cmapM(corr(:,1),:);
    for i=1:size(cmapN,1)
        if(cmapN(i,1) == 0 & cmapN(i,2) == 0 & cmapN(i,3) == 0)
            cmapN(i,:) = cmapM(corr_2(i,2),:);
        end
    end
    subplot(122), trisurf(N.TRIV, N.VERT(:,1), N.VERT(:,2), N.VERT(:,3), 1:N.n, 'EdgeAlpha', 0), axis equal, axis off;
    colormap(gca, cmapN);
end
