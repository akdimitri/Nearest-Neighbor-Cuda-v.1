%Matlab Script Q,C Sets Generation
%NN calculation
N_power = 21;
N = 2^N_power
C = rand(N,3);
Q = rand(N,3);
%Specify Indeces and shortest Distances
[Idx, D] = knnsearch( C, Q);

%Write Distances to files
fileID = fopen('./test/MatlabDistances21.txt','w');
fprintf( fileID,'%f\n', D);
fclose(fileID);

%Write Indeces to File
fileID = fopen('./test/MatlabIndeces21.txt','w');
fprintf( fileID,'%d\n', Idx);
fclose(fileID);

%Write Q to File
fileID = fopen('./test/MatlabQ21.txt','w');
for i = 1:N
fprintf( fileID,'%f %f %f\n', Q(i,1), Q(i,2), Q(i,3));
end
fclose(fileID);

%Write C to File
fileID = fopen('./test/MatlabC21.txt','w');
for i = 1:N
fprintf( fileID,'%f %f %f\n', C(i,1), C(i,2), C(i,3));
end
fclose(fileID);
