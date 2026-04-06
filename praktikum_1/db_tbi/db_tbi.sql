-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 24 Sep 2023 pada 02.56
-- Versi server: 10.4.21-MariaDB
-- Versi PHP: 8.0.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_tbi`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `dosen_pa`
--

CREATE TABLE `dosen_pa` (
  `id_dosen_pa` char(10) NOT NULL,
  `nama_dosen` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `dosen_pa`
--

INSERT INTO `dosen_pa` (`id_dosen_pa`, `nama_dosen`) VALUES
('0615128401', 'Fandy Setyo Utomo'),
('0617078601', 'Budi Rahardjo'),
('0618038201', 'Sanusi'),
('0623069001', 'Fitrah Rantika'),
('0630119201', 'Rina Andriani');

-- --------------------------------------------------------

--
-- Struktur dari tabel `mahasiswa`
--

CREATE TABLE `mahasiswa` (
  `nim` varchar(10) NOT NULL,
  `nama_mahasiswa` varchar(50) NOT NULL,
  `id_prodi` varchar(5) NOT NULL,
  `id_dosen_pa` char(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `mahasiswa`
--

INSERT INTO `mahasiswa` (`nim`, `nama_mahasiswa`, `id_prodi`, `id_dosen_pa`) VALUES
('20SA3150', 'Dewi', 'TI', '0630119201'),
('21SA1134', 'Yulian', 'IF', '0615128401'),
('21SA2123', 'Anton', 'SI', '0617078601'),
('22SA1987', 'Bayu', 'IF', '0615128401'),
('22SA2765', 'Ayu', 'SI', '0630119201');

-- --------------------------------------------------------

--
-- Struktur dari tabel `prodi`
--

CREATE TABLE `prodi` (
  `id_prodi` varchar(5) NOT NULL,
  `nama_prodi` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `prodi`
--

INSERT INTO `prodi` (`id_prodi`, `nama_prodi`) VALUES
('BD', 'Bisnis Digital'),
('IF', 'Informatika'),
('ILKOM', 'Ilmu Komunikasi'),
('SI', 'Sistem Informasi'),
('TI', 'Teknologi Informasi');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `dosen_pa`
--
ALTER TABLE `dosen_pa`
  ADD PRIMARY KEY (`id_dosen_pa`);

--
-- Indeks untuk tabel `mahasiswa`
--
ALTER TABLE `mahasiswa`
  ADD PRIMARY KEY (`nim`),
  ADD KEY `id_prodi` (`id_prodi`),
  ADD KEY `id_dosen_pa` (`id_dosen_pa`);

--
-- Indeks untuk tabel `prodi`
--
ALTER TABLE `prodi`
  ADD PRIMARY KEY (`id_prodi`);

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `mahasiswa`
--
ALTER TABLE `mahasiswa`
  ADD CONSTRAINT `mahasiswa_ibfk_1` FOREIGN KEY (`id_prodi`) REFERENCES `prodi` (`id_prodi`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `mahasiswa_ibfk_2` FOREIGN KEY (`id_dosen_pa`) REFERENCES `dosen_pa` (`id_dosen_pa`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
