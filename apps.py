import streamlit as st
import numpy as np

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

# Streamlit antarmuka
st.title("Kalkulator Dekomposisi Matriks")
st.header("Selamat Datang di Kalkulator Dekomposisi Crout & Doolittle by Kelompok 10 Kelas B")

matrix_input = st.text_area("Masukkan matriks Anda (dengan cara pisahkan elemen dengan spasi dan baris dengan baris baru/enter)")

metode = st.selectbox("Pilih metode dekomposisi", ["Crout", "Doolittle"])

if st.button("Dekomposisi"):
        A = text_to_matrix(matrix_input)
        if metode == "Crout":
            L, U = crout_decomposition(A)
            st.subheader('Crout Decomposition')
            st.write("Matrix L:", L)
            st.write("Matrix U:", U)
            st.write("L x U:", L @ U)
            
        elif metode == "Doolittle":
            L, U = doolittle_decomposition(A)
            st.subheader('Doolittle Decomposition')
            st.write("Matrix L =", L)
            st.write("Matrix U =", U)
            st.write("L x U =", L @ U)