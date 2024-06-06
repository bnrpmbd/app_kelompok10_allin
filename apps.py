import streamlit as st
import numpy as np
import scipy

def create_identity_matrix(size):
    return np.eye(size)

def subtract_matrices(A, B):
    return A - B

def determinant(matrix):
    return np.linalg.det(matrix)

def find_eigenvalues(A):
    return np.linalg.eigvals(A)

def find_eigenvectors(A):
    _, eigenvectors = np.linalg.eig(A)
    return eigenvectors

def diagonalize(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return P, D, P_inv

def lu_decomposition(A):
    P, L, U = scipy.linalg.lu(A)
    return P, L, U

def is_symmetric(A):
    return (A == A.T).all()

def make_symmetric(A):
    return (A + A.T) / 2

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def make_positive_definite(A):
    # Add a small positive value to diagonal elements
    min_eig = np.min(np.real(np.linalg.eigvals(A)))
    if min_eig <= 0:
        A += np.eye(A.shape[0]) * (1 - min_eig)

def cholesky_decomposition(A):
    try:
        L = np.linalg.cholesky(A)
        return L, L.T
    except np.linalg.LinAlgError:
        return None

# Fungsi dekomposisi Crout
def crout_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.double)
    U = np.eye(n, dtype=np.double)

    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))
        for i in range(j + 1, n):
            U[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(j))) / L[j, j]
            if np.isnan(U[j][i]):
                U[j][i] = 0
    
    return L, U

# Fungsi dekomposisi Doolitle
def doolittle_decomposition(A):
    n = A.shape[0]
    L = np.eye(n, dtype=np.double)
    U = np.zeros((n, n), dtype=np.double)

    for j in range(n):
        for i in range(j + 1):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for i in range(j, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))) / U[j, j]
            if np.isnan(L[i][j]):
                L[i][j] = 0

    return L, U

# Fungsi untuk mengubah input teks menjadi matriks numpy
def text_to_matrix(text):
    try:
        rows = text.strip().split('\n')
        matrix = [list(map(float, row.split())) for row in rows]
        return np.array(matrix)
    except Exception as e:
        st.error(f"Error converting input to matrix: {e}")
        return None

def main():
    st.title("Kalkulator Matriks")
    st.header("Selamat Datang di Kalkulator Matriks by Kelompok 10 Kelas B")

    matrix_input = st.text_area("Masukkan matriks Anda (dengan cara pisahkan elemen dengan spasi dan baris dengan baris baru/enter):", "1 -1 -9\n-1 3 -9\n1 -1 3")
    metode = st.selectbox("Pilih Metode Matriks", ["Nilai Eigen & Vektor Eigen", "Diagonalisasi", "Dekomposisi LU", "Dekomposisi Cholesky", "Dekomposisi Doolittle", "Dekomposisi Crout"])

    if st.button("HASIL"):
        if metode == "Nilai Eigen & Vektor Eigen":
            A = text_to_matrix(matrix_input)
            st.write("Input Matrix:")
            st.write(A)
            st.header("Nilai Eigen & Vektor Eigen")
            eigenvalues = find_eigenvalues(A)
            eigenvectors = find_eigenvectors(A)
            st.write("Nilai Eigen:")
            st.write(eigenvalues)
            st.write("Vektor Eigen:")
            st.write(eigenvectors)

        if metode == "Diagonalisasi":
            A = text_to_matrix(matrix_input)
            st.write("Input Matrix:")
            st.write(A)
            st.header("Diagonalisasi")
            try:
                P, D, P_inv = diagonalize(A)
                st.write("P (Vektor Eigen Matrix):")
                st.write(P)
                st.write("P_inv (Inverse of P):")
                st.write(P_inv)
                st.write("D (Diagonal Matrix of Eigenvalues):")
                st.write(D)
            except np.linalg.LinAlgError as e:
                st.write("Diagonalization failed:", e)

        if metode == "Dekomposisi LU":
            A = text_to_matrix(matrix_input)
            st.write("Input Matrix:")
            st.write(A)
            st.header("Dekomposisi LU")
            try:
                P, L, U = lu_decomposition(A)
                st.write("P (Permutation Matrix):")
                st.write(P)
                st.write("L (Lower Triangular Matrix):")
                st.write(L)
                st.write("U (Upper Triangular Matrix):")
                st.write(U)
                st.write("L x U:", L @ U)
            except np.linalg.LinAlgError as e:
                st.write("LU Decomposition failed:", e)
        if metode == "Dekomposisi Cholesky":
            try:
                A = text_to_matrix(matrix_input)
                if A.shape[0] != A.shape[1]:
                    st.error("Input matrix must be square.")
                else:
                    st.write("Input Matrix:")
                    st.write(A)
                    st.header("Dekomposisi Cholesky")
                    if not is_symmetric(A):
                        st.write("Matrix is not symmetric. Converting to symmetric matrix.")
                        A = make_symmetric(A)

                    if not is_positive_definite(A):
                        st.write("Matrix is not positive definite. Converting to positive definite matrix.")
                        make_positive_definite(A)

                    st.write("Processed Matrix:")
                    st.write(A)
                    L, L_T = cholesky_decomposition(A)
                    st.write("L (Lower Triangular Matrix):")
                    st.write(L)
                    st.write("L.T (Transpose of L):")
                    st.write(L_T)
                    st.write("L x L.T:", L @ L_T)
            except np.linalg.LinAlgError as e:
                st.write("Cholesky Decomposition failed:", e)
        
        if metode == "Dekomposisi Doolittle":
            A = text_to_matrix(matrix_input)
            st.write("Input Matrix:")
            st.write(A)
            st.header("Dekomposisi Doolittle")
            try:
                L, U = doolittle_decomposition(A)
                st.write("L (Lower Triangular Matrix):")
                st.write(L)
                st.write("U (Upper Triangular Matrix):")
                st.write(U)
                st.write("L x U:", L @ U)
            except np.linalg.LinAlgError as e:
                st.write("Doolittle Decomposition failed:", e)

        if metode == "Dekomposisi Crout":
            A = text_to_matrix(matrix_input)
            st.write("Input Matrix:")
            st.write(A)
            st.header("Dekomposisi Crout")
            try:
                L, U = crout_decomposition(A)
                st.write("L (Lower Triangular Matrix):")
                st.write(L)
                st.write("U (Upper Triangular Matrix):")
                st.write(U)
                st.write("L x U:", L @ U)
            except np.linalg.LinAlgError as e:
                st.write("Crout Decomposition failed:", e)

# Memanggil fungsi utama yang berisi streamlit antarmuka
if __name__ == "__main__":
    main()