#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <cmath>

using namespace cv;
using namespace std;

//Función que llena una matriz (extendida por el kernel) con 0's
Mat LlenadoCeros(Mat copia, int fila_original, int columna_original, int filas_copia, int columnas_copia, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;

	for (i = 0; i < filas_copia; i++) {
		for (j = 0; j < columnas_copia; j++) {
			if (i < filas_limite || i>fila_original + filas_limite) { //Rellena el extremo derecho/izquierdo de la imagen
				copia.at<uchar>(Point(j, i)) = 0; //Escribe 0 en el pixel i,j
			}
			if (j < columnas_limite || j > columna_original + columnas_limite) { //Rellena el extremo superior/inferior de la imagen
				copia.at<uchar>(Point(j, i)) = 0; //Escribe 0 en el pixel i,j
			}
		}
	}
	return copia;
}

//Función que pasa de color a escala de grises con NTSC
Mat RGB2Gray(Mat imagen, Mat Gray, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	double azul = 0;
	double verde = 0;
	double rojo = 0;
	double pixel = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			azul = imagen.at<Vec3b>(Point(i, j)).val[0]; //Obtiene valor del canal azul del pixel i,j
			verde = imagen.at<Vec3b>(Point(i, j)).val[1]; //Obtiene valor del canal verde del pixel i,j
			rojo = imagen.at<Vec3b>(Point(i, j)).val[2]; //Obtiene valor del canal rojo del pixel i,j
			pixel = (azul * 0.114 + verde * 0.587 + rojo * 0.299); //Saca el nivel de brillantez en grises de acuerdo al NTSC
			Gray.at<uchar>(Point(i, j)) = uchar(pixel); //Escribe el valor en la matriz Gray que es la imagen en grises
		}
	}
	return Gray;
}

//Función que genera la imagen con bordes ajustados de acuerdo al kernel
Mat AjusteBordes(Mat imagen, Mat copia, int fila_original, int columna_original, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;
	double azul = 0;
	double verde = 0;
	double rojo = 0;
	double pixel = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			azul = imagen.at<Vec3b>(Point(i, j)).val[0]; //Obtiene valor del canal azul del pixel i,j
			verde = imagen.at<Vec3b>(Point(i, j)).val[1];  //Obtiene valor del canal verde del pixel i,j
			rojo = imagen.at<Vec3b>(Point(i, j)).val[2]; //Obtiene valor del canal rojo del pixel i,j
			pixel = (azul * 0.114 + verde * 0.587 + rojo * 0.299); //Saca el nivel de brillantez en grises de acuerdo al NTSC
			copia.at<uchar>(Point(i + columnas_limite, j + filas_limite)) = uchar(pixel); //Escribe el valor dejando el borde del kernel con 0's
		}
	}
	return copia;
}

//Función que ajusta la imagen ecualizada con bordes
Mat AjusteBordesEcualizada(Mat ecualizada, Mat copia2, int fila_original, int columna_original, int filas_limite2, int columnas_limite2) {
	int i = 0;
	int j = 0;
	double pixel = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			pixel = ecualizada.at<uchar>(Point(i, j)); //Saca el nivel de brillantez en grises de acuerdo al NTSC
			copia2.at<uchar>(Point(i + columnas_limite2, j + filas_limite2)) = uchar(pixel); //Escribe el valor dejando el borde del kernel con 0's
		}
	}
	return copia2;
}

//Función que genera nuestro kernel de acuerdo con la formula de 
double** InicializaciónKernel(double** kernel, int size_kernel, int sigma, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;
	int x = 0;
	int y = 0;
	double pixel_kernel = 0;
	x = -(filas_limite);
	y = columnas_limite; //Comenzamos en la esquina superior izquierda

	for (i = 0; i < size_kernel; i++) {
		kernel[i] = new double[size_kernel]; //Crea la matriz dinamica del kernel
		for (j = 0; j < size_kernel; j++) {
			pixel_kernel = (1 / (2 * (3.1416) * (pow(sigma, 2))) * (exp(-(pow(x, 2) + (pow(y, 2))) / (2 * (pow(sigma, 2)))))); //Aplica la formula de Gauss
			kernel[i][j] = pixel_kernel; //Coloca el valor en el pixel i,j del kernel
			//cout << pixel_kernel << "   ";
			x = x + 1; //Vamos  avanzando en filas 
		}
		//cout << "\n";
		x = -(filas_limite);
		y = y - 1; //Vamos avanzando en columnas
	}
	return kernel;
}

//Función que realiza la multiplicación del pixel central del kernel por sus vecinos más cercanos
float Convolusion(Mat Gray, int extremos, double** kernel, int x, int y) {
	int i = 0;
	int j = 0;
	float valorPix = 0;

	for (i = -extremos; i <= extremos; i++) {
		for (j = -extremos; j <= extremos; j++) { //Recorreremos el kernel
			float valor_kernel = kernel[i + extremos][j + extremos]; //Obtenemos el valor del pixel 0,0
			int vecino_x = x + i; //Obtenemos coordenada de vecino en x
			int vecino_y = y + j; //Obtenemos coordenada de vecino en y
			float valor_img_original = 0;
			valor_img_original = Gray.at<uchar>(Point(vecino_x, vecino_y)); //Obtenemos el valor del vecino de la imagen en escala de grises
			valorPix = valorPix + (valor_kernel * valor_img_original); //Multiplicamos y vamos añadiendo a la suma de toda la vecindad del kernel
		}
	}
	return valorPix; //Regresamos el valor de la suma de la multiplicación del centro por cada vecino
}

//Función que aplica el filtro de Gauss a la imagen
Mat AplicaFiltroGauss(Mat Gray, Mat Gauss, double** kernel, int size_kernel, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	int extremos = floor(size_kernel / 2); //Obtenemos valor de las filas/columnas que rodearan el centro del kernel
	int l = 0;
	for (i = 0; i < fila_original; i++) { //Nos aseguramos de no poner en el centro del kernel un valor que no sea de la imagen a filtrar
		int k = 0;
		for (j = 0; j < columna_original; j++) { //Nos aseguramos de no poner en el centro del kernel un valor que no sea de la imagen a filtrar
			Gauss.at<uchar>(Point(l, k)) = uchar(Convolusion(Gray, extremos, kernel, i, j)); //Mandamos el valor del pixel que es el centro del kernel 
			k = k + 1; //Vamos avanzando en columnas
		}
		l = l + 1; //Vamos avanzando en filas
	}
	return Gauss;
}

//Función que multiplica el centro de la mascara Gx por cada vecino cercano
float Multiplicacion(Mat ecualizada, int extremos, int MascaraG[3][3], int x, int y) {
	int i = 0;
	int j = 0;
	float valorPix = 0;

	for (i = -extremos; i <= extremos; i++) {
		for (j = -extremos; j <= extremos; j++) {//REcorreremos la mascara completa
			float valor_mascara_g = MascaraG[i+extremos][j+extremos]; //Obtenemos el valor del pixel 0,0
			int vecino_x = x + i; //Obtenemos coordenada de vecino en x
			int vecino_y = y + j; //Obtenemos coordenada de vecino en y
			float valor_img_original = 0;
			valor_img_original = ecualizada.at<uchar>(Point(vecino_x, vecino_y)); //Obtenemos el valor del vecino de la imagen en escala de grises
			valorPix = valorPix + (valor_mascara_g * valor_img_original); //Multiplicamos y vamos añadiendo a la suma de toda la vecindad de la mascara
		}
	}
	return valorPix; //Regresamos el valor de la suma de la multiplicación del centro por cada vecino
}

//Función que multiplica la mascara Gx por la imagen ecualizada
Mat AplicaMascara(Mat G, Mat ecualizada, int MascaraG[3][3], int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	int extremos = floor(3/2); //Obtenemos valor de las filas/columnas que rodearan el centro de la mascara
	int l = 0;
	for (i = 0; i < fila_original; i++) { //Nos aseguramos de no poner en el centro de la mascara un valor que no sea de la imagen a filtrar
		int k = 0;
		for (j = 0; j < columna_original; j++) { //Nos aseguramos de no poner en el centro de la mascara un valor que no sea de la imagen a filtrar
			G.at<uchar>(Point(l, k)) = uchar(Multiplicacion(ecualizada, extremos, MascaraG, i, j)); //Mandamos el valor del pixel que es el centro de la mascara
			k = k + 1; //Vamos avanzando en columnas
		}
		l = l + 1; //Vamos avanzando en filas
	}
	return G;
}

//Función que obtiene la matriz |G|
Mat ObtenerAbsG(Mat G, Mat Gx, Mat Gy, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	int valorGx = 0;
	int valorGy = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			valorGx = abs(Gx.at<uchar>(Point(i, j))); //Obtenemos Gx(i,j)
			valorGy = abs(Gy.at<uchar>(Point(i, j))); //Obtenemos Gy(i,j)
			G.at<uchar>(Point(i, j)) = uchar(valorGx + valorGy); //Hacemos la suma |Gx| + |Gy|
		}
	}
	return G;
}

Mat AplicarCanny(Mat imagen, Mat CannyMat, int lim_inf, int lim_sup) {
	Canny(imagen, CannyMat, 0, 100); //No pude Marta :c
	return CannyMat;
}

//Función para obtener la orientación del gradiente
/*Mat ObtenerMagni(Mat Magnitud, Mat Gx, Mat Gy, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	int valorGx = 0;
	int valorGy = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			valorGx = Gx.at<uchar>(Point(i, j)); //Obtenemos Gx(i,j)
			valorGy = Gy.at<uchar>(Point(i, j)); //Obtenemos Gy(i,j)
			Magnitud.at<uchar>(Point(i, j)) = uchar(atan(valorGy / valorGx)); //Obtenemos la tan^-1(Gy/Gx)
		}
	}
	return Magnitud;
}*/

int main() {
	/*****DECLARACION DE LAS VARIABLES GENERALES***/
	char NombreImagen[] = "lena.png";
	Mat imagen; //Matriz que contiene nuestra imagen sin importar el formato
	/**********/

	/****LECTURA DE LA IMAGEN*****/
	imagen = imread(NombreImagen);

	if (!imagen.data) {
		cout << "ERROR AL CARGAR LA IMAGEN: " << NombreImagen << endl;
		exit(1);
	}

	/***PROCESOS****/
	int fila_original = imagen.rows; //Obtiene numero de filas de la imagen original
	int columna_original = imagen.cols; //Obtiene numero de columnas de la imagen original
	int size_kernel = 0;
	float sigma = 0;

	cout << "Ingresa kernel:\n"; //Pide el tamaño del kernel dinamico
	cin >> size_kernel;
	cout << "Ingresa sigma:\n"; //Pide el valor del sigma dinamico
	cin >> sigma;

	int filas_add = floor(size_kernel / 2); //Calcula el centro del kernel
	int filas_add_totales = filas_add * 2; //Calcula las filas totales a agregar
	int columnas_add_totales = filas_add_totales; //Calcula las columnas totales a agregar
	int filas_copia = filas_add_totales + fila_original; //Calcula el tamaño de la imagen (en filas) ya con bordes
	int columnas_copia = columnas_add_totales + columna_original; //Calcula el tamaño de la imagen (en columnas) ya con bordes

	Mat Gray(fila_original, columna_original, CV_8UC1); //Genera la matriz para la imagen a escala de grises
	Gray = RGB2Gray(imagen, Gray, fila_original, columna_original); //Pasa la imagen a escala de grises

	int filas_limite = floor(size_kernel / 2); //Calcula las filas 
	int columnas_limite = filas_limite;
	Mat copia(filas_copia, columnas_copia, CV_8UC1); //Genera la matriz para la imagen con bordes
	copia = LlenadoCeros(copia, fila_original, columna_original, filas_copia, columnas_copia, filas_limite, columnas_limite); //Rellena el contorno con 0's
	copia = AjusteBordes(imagen, copia, fila_original, columna_original, filas_limite, columnas_limite); //Rellena la imagen en el centro del marco de bordes con 0's

	double** kernel = new double* [size_kernel]; //Genera el kernel
	kernel = InicializaciónKernel(kernel, size_kernel, sigma, filas_limite, columnas_limite);

	Mat Gauss(fila_original, columna_original, CV_8UC1); //Genera la matriz para la imagen con el filtro de Gauss
	Gauss = AplicaFiltroGauss(Gray, Gauss, kernel, size_kernel, fila_original, columna_original); //Aplica el filtro de Gauss sobre la imagen en escala de grises

	Mat ecualizada(fila_original, columna_original, CV_8UC1); //Genera la matriz para la imagen ecualizada
	equalizeHist(Gauss, ecualizada); //Ecualiza la imagen obtenida de pasar el filtro de Gauss

	int filas_copia2 = fila_original + 2; //Agregamos 2 filas ya que nuestra mascara es de 3x3
	int columnas_copia2 = columna_original + 2; //Agregamos 2 columnas ya que nuestra mascara es de 3x3
	int filas_limite2 = floor(3 / 2); //Calcula las filas a agregar por lado 
	int columnas_limite2 = filas_limite2; //Calcula las columnas a agregar por lado

	Mat copia2(filas_copia2, columnas_copia2, CV_8UC1); //Genera la matriz para la imagen con bordes
	copia2 = LlenadoCeros(copia2, fila_original, columna_original, filas_copia2, columnas_copia2, filas_limite2, columnas_limite2); //Rellena el contorno con 0's
	copia2 = AjusteBordesEcualizada(ecualizada, copia2, fila_original, columna_original, filas_limite2, columnas_limite2); //Obtiene la imagen ecualizada con bordes

	int MascaraGx[3][3]; //Genera la mascara Gx
	MascaraGx[0][0] = -1;
	MascaraGx[0][1] = 0;
	MascaraGx[0][2] = 1;
	MascaraGx[1][0] = -2;
	MascaraGx[1][1] = 0;
	MascaraGx[1][2] = 2;
	MascaraGx[2][0] = -1;
	MascaraGx[2][1] = 0;
	MascaraGx[2][2] = 1;

	Mat Gx(fila_original, columna_original, CV_8UC1); //Genera la matriz para obtener los coeficientes de Gx con Sobel
	Gx = AplicaMascara(Gx, copia2, MascaraGx, fila_original, columna_original); //Obtenemos la matriz Gx

	int MascaraGy[3][3]; //Genera la mascara Gy
	MascaraGy[0][0] = -1;
	MascaraGy[0][1] = -2;
	MascaraGy[0][2] = -1;
	MascaraGy[1][0] = 0;
	MascaraGy[1][1] = 0;
	MascaraGy[1][2] = 0;
	MascaraGy[2][0] = 1;
	MascaraGy[2][1] = 2;
	MascaraGy[2][2] = 1;

	Mat Gy(fila_original, columna_original, CV_8UC1); //Genera la matriz para obtener los coeficientes de Gy con Sobel
	Gy = AplicaMascara(Gy, copia2, MascaraGy, fila_original, columna_original); //Obtenemos la matriz Gy

	Mat G(fila_original, columna_original, CV_8UC1); //Genera la matriz para obtener los coeficientes de |G| con Sobel
	G = ObtenerAbsG(G, Gx, Gy, fila_original, columna_original);

	Mat CannyMat(fila_original, columna_original, CV_8UC1);
	CannyMat=AplicarCanny(imagen, CannyMat, 0,100);
	//Magnitud = ObtenerMagni(Magnitud,Gx,Gy,fila_original,columna_original);

	namedWindow("Original", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Original", imagen); //Muestra la imagen original

	namedWindow("Grises", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Grises", Gray); //Muestra la imagen en escala de grises

	namedWindow("Gauss", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Gauss", Gauss); //Muestra la imagen con el filtro de Gauss

	namedWindow("Ecualizada", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Ecualizada", ecualizada); //Muestra la imagen ecualizada

	namedWindow("Img transicion Gx", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Img transicion Gx", Gx); //Muestra la imagen Gx

	namedWindow("Img transicion Gy", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Img transicion Gy", Gy); //Muestra la imagen Gy

	namedWindow("Sobel", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Sobel", G); //Muestra la imagen |G|

	namedWindow("Canny", WINDOW_AUTOSIZE); //Creacion de una ventana
	imshow("Canny", CannyMat); //Muestra la imagen de magnitud

	cout << "Tamaño del kernel usado para Gauss:" << size_kernel << "x" << size_kernel << "\n\n"; //Muestra el tamaño del kernel
	cout << "Filas Imagen Original: " << fila_original << endl; //Muestra tamaño en filas de la imagen original
	cout << "Columnas Imagen Original: " << columna_original << "\n\n"; //Muestra tamaño en columnas de la imagen original
	cout << "Filas Imagen Grises: " << Gray.rows << endl; //Muestra tamaño en filas de la imagen en escala de grises
	cout << "Columnas Imagen Grises: " << Gray.cols << "\n\n"; //Muestra tamaño en columnas de la imagen en escala de grises
	cout << "Filas Imagen con Filtro de Gauss: " << Gauss.rows << endl; //Muestra tamaño en filas de la imagen con filtro de Gauss
	cout << "Columnas Imagen con Filtro de Gauss: " << Gauss.cols << "\n\n"; //Muestra tamaño en columnas de la imagen con filtro de Gauss
	cout << "Filas Imagen Ecualizada: " << ecualizada.rows << endl; //Muestra tamaño en filas de la imagen ecualizada
	cout << "Columnas Imagen Ecualizada: " << ecualizada.cols << "\n\n"; //Muestra tamaño en columnas de la imagen ecualizada
	cout << "Filas Imagen Gx: " << Gx.rows << endl; //Muestra tamaño en filas de la imagen Gx
	cout << "Columnas Imagen Gx: " << Gx.cols << "\n\n"; //Muestra tamaño en columnas de la imagen Gx
	cout << "Filas Imagen Gy: " << Gy.rows << endl; //Muestra tamaño en filas de la imagen Gy
	cout << "Columnas Imagen Gy: " << Gy.cols << "\n\n"; //Muestra tamaño en columnas de la imagen Gy
	cout << "Filas Imagen |G|: " << G.rows << endl; //Muestra tamaño en filas de la imagen |G|
	cout << "Columnas Imagen |G|: " << G.cols << "\n\n"; //Muestra tamaño en columnas de la imagen |G|

	waitKey(0);
	return 1;
}