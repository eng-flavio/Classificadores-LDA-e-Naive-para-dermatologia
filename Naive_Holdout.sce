clc
clear 
Base = csvRead('dermatology.csv', ',');

//Remoção de amostras incompletas
for i=1:4
    Base(263,:) = [];
end

for i=1:4
    Base(34,:) = [];
end

//Definição de parâmetros
amostras=358;
atributos = 34;
decisao = zeros(1,6);

//Z-score
Atributos_z = zeros(amostras,34);
 
for i=1:amostras
    for k = 1: atributos
        Atributos_z(i,k) = (Base(i,k)-mean(Base(:,k)))/stdev(Base(:,k));
    end 
end
Base(:,1:34) = Atributos_z;

//Houldout
count = 0;
for nn=1:20

//randomiza as linhas da matriz
[m,n] = size(Base);
grand(1, "prm", (1:amostras)')' ;
v = grand(1, "prm", (1:amostras)')'; 
idx  = v(1:amostras);
nova_base = zeros(amostras,35);

for i=1:amostras
    nova_base(i,:) = Base(v(i),:);
end
Base = nova_base;
//Separação de Rotulos e Atributos
Treino_amostras = Base(21:amostras,1:34);
Treino_rotulos = Base(21:amostras,35);
Teste_amostras = Base(1:20,1:34);
Teste_rotulos = Base(1:20,35);



///////////////////////////////////LDA//////////////////////////////////////////
// Separada as classes em matrizes 
count1 = 0 ;
count2 = 0 ;
count3 = 0 ;
count4 = 0 ;
count5 = 0 ;
count6 = 0 ;

Atributos = Treino_amostras
Rotulos = Treino_rotulos

for i=1:amostras-20
    if Rotulos(i) == 1
        count1 = count1 + 1;
    end
    if Rotulos(i) == 2
        count2 = count3 + 1;
    end
    if Rotulos(i) == 3
        count3 = count3 + 1;
    end
    if Rotulos(i) == 4
        count4 = count4 + 1;
    end
    if Rotulos(i) == 5
        count5 = count5 + 1;
    end
    if Rotulos(i) == 6
        count6 = count6 + 1;
    end
end

X1 = zeros(count1,34);
X2 = zeros(count2,34);
X3 = zeros(count3,34);
X4 = zeros(count4,34);
X5 = zeros(count5,34);
X6 = zeros(count6,34);


count1 = 0 ;
count2 = 0 ;
count3 = 0 ;
count4 = 0 ;
count5 = 0 ;
count6 = 0 ;


for i=1:amostras-20
    if Rotulos(i) == 1
        count1 = count1 +1;
        X1(count1,:) = Atributos(i,:);
    end
    if Rotulos(i) == 2
        count2 = count2 +1;
        X2(count2,:) = Atributos(i,:);
    end
    if Rotulos(i) == 3
        count3 = count3 +1;
        X3(count3,:) = Atributos(i,:);
    end
    if Rotulos(i) == 4
        count4 = count4 +1;
        X4(count4,:) = Atributos(i,:);
    end
    if Rotulos(i) == 5
        count5 = count5 +1;
        X5(count5,:) = Atributos(i,:);
    end
    if Rotulos(i) == 6
        count6 = count6 +1;
        X6(count6,:) = Atributos(i,:);
    end
end


// Média dos atributos
u = mean(Atributos,'r')';
u1 = mean(X1,'r')';
u2 = mean(X2,'r')';
u3 = mean(X3,'r')';
u4 = mean(X4,'r')';
u5 = mean(X5,'r')';
u6 = mean(X6,'r')';


// Matriz de covariância
S1 = cov(X1);
S2 = cov(X2);
S3 = cov(X3);
S4 = cov(X4);
S5 = cov(X5);
S6 = cov(X6);

// Deixando apenas a diagonal principal com valores não-nulos
for i=1:34
    for k=1:34
        if k ~= i
            S1(k,i) = 0;
            S2(k,i) = 0;
            S3(k,i) = 0;
            S4(k,i) = 0;
            S5(k,i) = 0;
            S6(k,i) = 0;
        end
    end
end
// Within-class scatter matrix
Sw = S1+S2+S3+S4+S5+S6;

// Between-class scatter matrix
[A1,B1] = size(X1);
[A2,B2] = size(X2);
[A3,B3] = size(X3);
[A4,B4] = size(X4);
[A5,B5] = size(X5);
[A6,B6] = size(X6);

SB = (A1*(u1 - u)*(u1 - u)'+A2*(u2 - u)*(u2 - u)'+A3*(u3 - u)*(u3 - u)'+A4*(u4 - u)*(u4 - u)'+A5*(u5 - u)*(u5 - u)'+A6*(u6 - u)*(u6 - u)');

// Eigendecomposition 
[V,D] = spec(inv(Sw)*SB);

// Vetor de projeção ótimo
w = V(:,1);
w=w';
// Vetor de projeção ótimo usando a fórmula
// w = inv(Sw)*(u1 - u2);
// w = -w/norm(w)


// Amostras projetadas
y1 = w*X1';
y2 = w*X2';
y3 = w*X3';
y4 = w*X4';
y5 = w*X5';
y6 = w*X6';


//Região de Decisão
decisao(1)=mean(y1);
decisao(2)=mean(y2);
decisao(3)=mean(y3);
decisao(4)=mean(y4);
decisao(5)=mean(y5);
decisão(6)=mean(y6);

// Testes
tm = w*Teste_amostras';
// Rotulação

Aux = zeros(6);
Rotulos_finais = zeros(20,1);

for i=1:20
    for k=1:6
        Aux(k) = abs(abs(tm(i))-abs(decisao(k)));        
    end
    [A,B] = min(Aux);
    Rotulos_finais(i) = B;
end

//Classificação

for i=1:20
    if Rotulos_finais(i) == Teste_rotulos(i);
        count = count +1;
    end
end
end
Resultado = count/400
