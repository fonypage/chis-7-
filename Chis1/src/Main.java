import java.io.*;
import java.util.*;

public class Main {
    private static final double EPSILON = 1e-6;

    public static void main(String[] args) {
        String inputFile = "input.txt";
        String outputFile = "output.txt";

        try {
            double[][] A = readMatrixFromFile(inputFile);
            int n = A.length;

            double[] x = new double[n];
            Arrays.fill(x, 1.0);

            double lambda = rayleighQuotient(A, x);
            double lambdaPrev, lambdaAitken = lambda;

            int iterations = 0;
            do {
                lambdaPrev = lambda;
                x = multiplyMatrixVector(A, x);
                normalize(x);

                double lambda1 = rayleighQuotient(A, x);
                double lambda2 = rayleighQuotient(A, multiplyMatrixVector(A, x));

                lambdaAitken = lambda1 - Math.pow(lambda2 - lambda1, 2) / (lambda2 - 2 * lambda1 + lambdaPrev);

                lambda = lambda1;
                iterations++;

            } while (Math.abs(lambda - lambdaPrev) > EPSILON);

            writeResultsToFile(outputFile, lambdaAitken, x, iterations);

        } catch (IOException e) {
            System.err.println("Ошибка при работе с файлами: " + e.getMessage());
        }
    }

    private static double rayleighQuotient(double[][] A, double[] x) {
        double[] Ax = multiplyMatrixVector(A, x);
        return dotProduct(Ax, x) / dotProduct(x, x);
    }

    private static double[] multiplyMatrixVector(double[][] A, double[] x) {
        int n = A.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i] += A[i][j] * x[j];
            }
        }
        return result;
    }

    private static void normalize(double[] x) {
        double norm = Math.sqrt(dotProduct(x, x));
        for (int i = 0; i < x.length; i++) {
            x[i] /= norm;
        }
    }

    private static double dotProduct(double[] v1, double[] v2) {
        double sum = 0;
        for (int i = 0; i < v1.length; i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    private static double[][] readMatrixFromFile(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        int n = Integer.parseInt(br.readLine().trim());
        double[][] A = new double[n][n];

        for (int i = 0; i < n; i++) {
            String[] line = br.readLine().trim().split("\\s+");
            for (int j = 0; j < n; j++) {
                A[i][j] = Double.parseDouble(line[j]);
            }
        }
        br.close();
        return A;
    }

    private static void writeResultsToFile(String filename, double lambda, double[] x, int iterations) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
        bw.write("Собственное значение (ускоренное): " + lambda + "\n");
        bw.write("Собственный вектор:\n");
        for (double v : x) {
            bw.write(v + " ");
        }
        bw.write("\nИтераций: " + iterations + "\n");
        bw.close();
    }
}