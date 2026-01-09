clear;
% curve shortening
dt = 0.01;
convergence_thresh = 1e-8;

% momentum primal
beta = .75;%0.999;

% momentum dual
dual_gain = 2.; %5.0;%1.75;      % scale on the dual step, try 1.0–2.0
eps_dual  = .1; %1.;%0.05;     % small derivative correction, try 0–0.1

%% Generate curve
N = 35;%32;
x = randn(N,1)-0.5;
y = randn(N,1)-0.5;
z = randn(N,1)-0.5;
gamma_rand = [x,y,z];

%initialize new init curve
theta = linspace(0,2*pi,N+1)'; theta = theta(1:N);
x = cos(theta);
y = sin(theta);
z = 0*theta;

gamma = [x,y,z];

gamma = gamma + 0.1 * gamma_rand;

N_experiments = 240;
eps_tmp = 1e-3;

% Sample interval in log-scale
tmp = linspace(log(eps_tmp), log(1000*pi + eps_tmp), N_experiments);
V_vals = exp(tmp) - eps_tmp;
% V_vals = linspace(1e-2,75*pi, N_experiments);

%% Loop thorugh a number of experiments
for val = 1:N_experiments

    
    %load('curve_42.mat');
    %gamma = [xx,yy,zz];
    %N = length(xx);
    
    %% Specify constraints
    V = [ 0. , 0.,  V_vals(val) ];
    A = [ -pi, 0., pi ] ;% / norm(V) ;
    %[A,V] = ComputeAreaVolume(xx,yy,zz);
    lambda_area = [0,0,0];
    dlambda_area = [0,0,0];
    lambda_vol = [0,0,0];
    dlambda_vol = [0,0,0];
    
    % adaptive step size
    %dt = ( dt * norm(A) ) / ( 1 + dot( A/norm(A) , V) );
    
    %% Prepare indices
    I = (1:N)';
    Ip = [(2:N)';1];
    Im = [N;(1:N-1)'];
    
    %% Laplacian
    L_row = [I;I;I];
    L_col = [I;Ip;Im];
    L_val = [2*ones(N,1);-1*ones(N,1);-1*ones(N,1)];
    L = sparse(L_row,L_col,L_val,N,N);
    M = speye(N);
    
    x = gamma(:,1);
    y = gamma(:,2);
    z = gamma(:,3);
    plotted_line = plot3([x;x(1)],[y;y(1)],[z;z(1)],'.-');
    
    
    grid on
    axis equal
    cameratoolbar
    drawnow
    
    %% Main iteration
    maxiter = 1000000;
    % maxiter = 25000;
    % maxiter = 15000;
    % maxiter = 10;
    
    % Area or Area+Volume Constraints
    volume_calc = true;
    
    % Momentum added to primal
    add_primal_momentum = true;
    
    % Momentum added to dual
    add_dual_momentum = true;
    
    % previous residuals for derivative correction
    G_area_prev = [0, 0, 0];
    G_vol_prev  = [0, 0, 0];
    
    % Vectorized volume gradient
    vectorize_gradients = true;
    
    gamma_old = zeros(size(gamma));
    
    gamma_prev = gamma;
    
    for iter = 1:maxiter
        % Decide extrapolated configuration
        if (add_primal_momentum == false)
            % no momentum
            gamma_tilde = gamma;
        else
            % momentum step
            gamma_tilde = gamma + beta * (gamma - gamma_prev);
        end
    
        % This is the configuration at which we evaluate gradients
        gamma_p = gamma_tilde;
    
        % update history for next iteration
        gamma_prev = gamma;
    
        % Neighbors
        gamma_Im = gamma_p(Im,:);
        gamma_I  = gamma_p(I,:);
        gamma_Ip = gamma_p(Ip,:);
    
        % Edge vectors for area gradient (note: choose one convention and stick to it)
        edge2 = gamma_Im - gamma_Ip;
    
        % Basis vectors
        ex = [1, 0, 0];
        ey = [0, 1, 0];
        ez = [0, 0, 1];
    
        Ex = repmat(ex, N, 1);
        Ey = repmat(ey, N, 1);
        Ez = repmat(ez, N, 1);
    
        %% Area gradient at gamma_tilde
        grad_Ax_gamma = cross(Ex, edge2, 2);
        grad_Ay_gamma = cross(Ey, edge2, 2);
        grad_Az_gamma = cross(Ez, edge2, 2);
    
        %% Volume gradient at gamma_tilde
        area_cross_m = cross(gamma_Im, gamma_I,  2);
        area_cross_p = cross(gamma_I,  gamma_Ip, 2);
        midpoints_m  = 0.5 * (gamma_Im + gamma_I);
        midpoints_p  = 0.5 * (gamma_I  + gamma_Ip);
    
        gamma_p_m = gamma_Im;
        gamma_p_p = gamma_Ip;
    
        % x components
        first_term_p_x = -(1/6) * cross(Ex, area_cross_m, 2);
        first_term_m_x = -(1/6) * cross(Ex, area_cross_p, 2);
    
        second_term_m_x_tmp = cross(Ex, midpoints_m, 2);
        second_term_p_x_tmp = cross(Ex, midpoints_p, 2);
    
        second_term_m_x =  (1/3) * cross(second_term_m_x_tmp, gamma_p_m, 2);
        second_term_p_x = -(1/3) * cross(second_term_p_x_tmp, gamma_p_p, 2);
    
        grad_Vx_gamma = first_term_m_x + first_term_p_x + second_term_m_x + second_term_p_x;
    
        % y components
        first_term_p_y = -(1/6) * cross(Ey, area_cross_m, 2);
        first_term_m_y = -(1/6) * cross(Ey, area_cross_p, 2);
    
        second_term_m_y_tmp = cross(Ey, midpoints_m, 2);
        second_term_p_y_tmp = cross(Ey, midpoints_p, 2);
    
        second_term_m_y =  (1/3) * cross(second_term_m_y_tmp, gamma_p_m, 2);
        second_term_p_y = -(1/3) * cross(second_term_p_y_tmp, gamma_p_p, 2);
    
        grad_Vy_gamma = first_term_m_y + first_term_p_y + second_term_m_y + second_term_p_y;
    
        % z components
        first_term_p_z = -(1/6) * cross(Ez, area_cross_m, 2);
        first_term_m_z = -(1/6) * cross(Ez, area_cross_p, 2);
    
        second_term_m_z_tmp = cross(Ez, midpoints_m, 2);
        second_term_p_z_tmp = cross(Ez, midpoints_p, 2);
    
        second_term_m_z =  (1/3) * cross(second_term_m_z_tmp, gamma_p_m, 2);
        second_term_p_z = -(1/3) * cross(second_term_p_z_tmp, gamma_p_p, 2);
    
        grad_Vz_gamma = first_term_m_z + first_term_p_z + second_term_m_z + second_term_p_z;
    
        %% Build gradient of the augmented Lagrangian
        if (volume_calc == false)
            g = ((lambda_area(1) + dlambda_area(1)) * grad_Ax_gamma) + ...
                ((lambda_area(2) + dlambda_area(2)) * grad_Ay_gamma) + ...
                ((lambda_area(3) + dlambda_area(3)) * grad_Az_gamma);
        else
            g = ((lambda_area(1) + dlambda_area(1)) * grad_Ax_gamma) + ...
                ((lambda_area(2) + dlambda_area(2)) * grad_Ay_gamma) + ...
                ((lambda_area(3) + dlambda_area(3)) * grad_Az_gamma) + ...
                ((lambda_vol(1)  + dlambda_vol(1))  * grad_Vx_gamma) + ...
                ((lambda_vol(2)  + dlambda_vol(2))  * grad_Vy_gamma) + ...
                ((lambda_vol(3)  + dlambda_vol(3))  * grad_Vz_gamma);
        end
    
        %% Preconditioned step
        rhs   = gamma_tilde - dt * g;
        gamma = (M + dt*L) \ rhs;
    
        %% Constraints and multipliers (as before, evaluated at gamma)
        % areas_cross_tri = cross(gamma(I,:), gamma(Ip,:), 2);
        % midpoints       = 0.5*(gamma(I, :) + gamma(Ip, :));
        % area            = 0.5*sum(areas_cross_tri);
        % vol             = (1/3) * sum(cross(midpoints, areas_cross_tri, 2));
    
        % G_area       = area - A;
        % dlambda_area = dt*G_area;
        % lambda_area  = lambda_area + dlambda_area;
    
        % G_vol        = vol - V;
        % dlambda_vol  = dt*G_vol;
        % lambda_vol   = lambda_vol + dlambda_vol;
        
        %% Compute corresponding area/volume constraints and lambda
        areas_cross_tri = cross(gamma(I,:), gamma(Ip,:), 2);
        midpoints       = 0.5*(gamma(I, :) + gamma(Ip, :));
        area            = 0.5*sum(areas_cross_tri);
        vol             = (1/3) * sum(cross(midpoints, areas_cross_tri, 2));
    
        G_area = area - A;          % 1x3
        G_vol  = vol  - V;          % 1x3
    
        if (add_dual_momentum == false)
            dlambda_area = dt*G_area;
            lambda_area  = lambda_area + dlambda_area;
        
            dlambda_vol  = dt*G_vol;
            lambda_vol   = lambda_vol + dlambda_vol;
       else
            % dual update with gain and derivative correction
            dlambda_area = dual_gain * dt * G_area ...
                         + eps_dual * (G_area - G_area_prev);
        
            dlambda_vol  = dual_gain * dt * G_vol  ...
                         + eps_dual * (G_vol  - G_vol_prev);
        
            lambda_area = lambda_area + dlambda_area;
            lambda_vol  = lambda_vol  + dlambda_vol;
        
            % store residuals for next iteration
            G_area_prev = G_area;
            G_vol_prev  = G_vol;
        end
        
    
        %% plotting etc. unchanged
        if mod(iter, 250) == 0
            x = gamma(:,1); y = gamma(:,2); z = gamma(:,3);
            plotted_line.XData = [x;x(1)];
            plotted_line.YData = [y;y(1)];
            plotted_line.ZData = [z;z(1)];
            axis equal
            title({ ...
                ['iter = ', num2str(iter), ' (maxiter = ', num2str(maxiter), ')', '  |  experiment = ', num2str(val), ' (N_experiments = ', num2str(N_experiments), ')'], ...
                ['G_{area,x} = ', num2str(G_area(1)), ...
                 '   G_{area,y} = ', num2str(G_area(2)), ...
                 '   G_{area,z} = ', num2str(G_area(3))], ...
                 ['G_{vol,x} = ', num2str(G_vol(1)), ...
                 '   G_{vol,y} = ', num2str(G_vol(2)), ...
                 '   G_{vol,z} = ', num2str(G_vol(3))] ...
            });
            drawnow
        end

        error = norm(G_area) + norm(G_vol);
        if (error < convergence_thresh)
             gamma_renormalized = gamma * norm(V) / norm(A);
             save_polyline_obj(gamma_renormalized , val);
             break
        end
    end
end


%% Gradient Calculation

% COME UP WITH SOME NEW MATRIX/VECTOR FORMULATION...  (x^T)Ax, x is
% 3*3x1 vector/ 3x3 matrix (or N instead of 3, if 3 doesn't represent
% (i-1,i,i+1)


% Input vector 
function cross_matrix = cross_vec2mat(vec)
    cross_matrix = [0, -vec(3), vec(2);
                    vec(3), 0, -vec(1);
                    -vec(2), vec(1), 0];
%    cross_matrix = cross_matrix';
end


function save_polyline_obj(gamma, iter)
% save_polyline_obj  Write a closed polyline OBJ for iteration "iter".
%
%   gamma : N-by-3 array of vertex positions
%   iter  : iteration number used in the filename

    % folder structure relative to this file
    basepath = fileparts(mfilename('fullpath'));
    export_folder = fullfile(basepath, 'export', 'LogScale_1e-2_to_25PI_renorm');

    if ~exist(export_folder, 'dir')
        mkdir(export_folder);
    end

    % close the loop
    X = [gamma(:,1); gamma(1,1)];
    Y = [gamma(:,2); gamma(1,2)];
    Z = [gamma(:,3); gamma(1,3)];
    N = numel(X);

    % file name
    fname = fullfile(export_folder, sprintf('polyline_iter%d.obj', iter));
    fid = fopen(fname, 'w');

    % vertices
    for k = 1:N
        fprintf(fid, 'v %.8f %.8f %.8f\n', X(k), Y(k), Z(k));
    end

    % polyline element
    fprintf(fid, 'l');
    for k = 1:N
        fprintf(fid, ' %d', k);
    end
    fprintf(fid, '\n');

    fclose(fid);
end