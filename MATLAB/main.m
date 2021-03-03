
clear all; close all; clc; workspace;


cd( fileparts( matlab.desktop.editor.getActiveFilename ) );                % Setting the current directory (cd) as the current location of this folder. 


myFigureConfig( 'fontsize', 20, ...
               'lineWidth',  5, ...
              'markerSize', 25    )    

%%

syms m k s

m = 1; k = 1;

Zm = m * s;
Zk = k / s;

Z0 = Zm;

Ztot = Z0;
for i = 1 : 3
    Ztot =  Zm + Zk * Ztot / ( Zk + Ztot );
    
end

Ztot = simplify( subs( Ztot, {m,k}, {1,1} )  );

% h=matlabFunction( ilaplace( Ztot, 's') );
% t=1:0.01:100;
% plot(t,h(t))



%%

data = myTxtParse( 'data_log.txt' )
