!==============================================================================
! print_rho_g.f90
!
! Reads a Quantum ESPRESSO charge-density file and prints all Fourier
! coefficients rho(G) to standard output, one line per G-vector:
!
!   rho(r) = (1/Omega) * sum_G rho_G * exp(i G . r)
!
! The charge density is located via QE's standard save-directory convention:
!   <outdir>/<prefix>.save/data-file-schema.xml   -- structure, FFT grid, ...
!   <outdir>/<prefix>.save/charge-density.dat     -- real-space density
!   (charge-density.hdf5 is used automatically when QE was compiled with HDF5
!    support; read_file() selects the correct format at runtime with no change
!    required here)
!
! QE modules used
! ---------------
!   environment    : environment_start / environment_end  (QE runtime setup)
!   mp_global      : mp_startup   (MPI initialisation)
!   mp_world       : world_comm   (global MPI communicator)
!   mp             : mp_bcast     (broadcast namelist to all MPI tasks)
!   io_global      : stdout, ionode, ionode_id  (serial I/O control)
!   io_files       : prefix, outdir, tmp_dir    (QE file-name variables)
!   cell_base      : tpiba = 2*pi/a0            (reciprocal-lattice scale)
!   gvect          : ngm, g(3,ngm), mill(3,ngm) (local G-vector list)
!   fft_base       : dfftp  (dense FFT descriptor; nl(:) maps G->grid index)
!   fft_interfaces : fwfft  (in-place forward FFT: real space -> G space)
!   scf            : rho    (rho%of_r is populated by read_file())
!   lsda_mod       : nspin  (number of spin components)
!   kinds          : DP     (double-precision kind parameter)
!
! Compilation
! -----------
! This file must be linked against QE's libraries.  With a QE 7.x build
! installed under $QE_PREFIX the compilation looks roughly like:
!
!   mpif90 -I$QE_PREFIX/include \
!          -o print_rho_g.x print_rho_g.f90 \
!          -L$QE_PREFIX/lib \
!          -lqemods -lpw -lks_solvers -ldevXlib -lUtilXlib \
!          -lFFTXlib -lLAXlib -lupflib \
!          -llapack -lblas
!
! The exact library list depends on the QE version and build flags.  The
! most reliable approach is to add this file as an executable target in
! QE's own CMake/configure build and let it inherit the link line
! automatically (see PW/CMakeLists.txt for a reference target).
!
! Usage
! -----
!   echo "&inputpp prefix='pwscf', outdir='./' /" | ./print_rho_g.x
!
!   # or using a namelist file:
!   ./print_rho_g.x < input.nml
!
!   # where input.nml contains:
!   #   &inputpp
!   #     prefix = 'myrun',
!   #     outdir = '/path/to/qe_output/'
!   #   /
!
! Output columns
! --------------
!   m1   m2   m3     Gx(1/Bohr)  Gy(1/Bohr)  Gz(1/Bohr)  Re(rho_G)  Im(rho_G)
!
! Normalisation note
! ------------------
! After fwfft the stored value is the discrete sum
!   psic(nl(ig)) = sum_{r on grid} rho(r) * exp(-i G.r)
! To obtain the physical coefficient rho_G = (1/Omega) * integral rho(r)
! exp(-iGr) d^3r, divide the printed values by nr1*nr2*nr3 (total number
! of real-space grid points).  Which convention is appropriate depends on
! the downstream application.
!
! MPI note
! --------
! QE distributes G-vectors and the FFT grid across MPI tasks.  This
! program assumes a single-process (serial) run: the ionode owns all ngm
! G-vectors and the full FFT grid.  For a parallel run, the G-vectors
! would need to be gathered from all tasks first (see gvect%ig_l2g for
! the local-to-global index mapping).  The program checks nprocs at
! runtime and aborts with an informative message if nprocs > 1.
!==============================================================================

PROGRAM print_rho_g

  ! QE runtime and I/O setup
  USE environment,    ONLY: environment_start, environment_end
  USE mp_global,      ONLY: mp_startup
  USE mp_world,       ONLY: world_comm, nprocs
  USE mp,             ONLY: mp_bcast
  USE io_global,      ONLY: stdout, ionode, ionode_id
  USE io_files,       ONLY: prefix, outdir, tmp_dir

  ! Lattice, reciprocal-lattice vectors, and FFT infrastructure
  USE cell_base,      ONLY: tpiba           ! 2*pi/a0, converts g to 1/Bohr
  USE gvect,          ONLY: ngm, g, mill    ! G-vector count, coordinates, Miller indices
  USE fft_base,       ONLY: dfftp           ! dense FFT descriptor (nr1,nr2,nr3,nl,...)
  USE fft_interfaces, ONLY: fwfft           ! forward FFT: real space -> G space

  ! Charge density and spin
  USE scf,            ONLY: rho             ! rho%of_r(:,ispin) after read_file()
  USE lsda_mod,       ONLY: nspin           ! 1 (non-spin-polarised) or 2 (LSDA)

  USE kinds,          ONLY: DP

  IMPLICIT NONE

  INTEGER               :: ispin, ig, ios
  COMPLEX(DP), ALLOCATABLE :: psic(:)       ! complex work array for FFT

  ! Namelist: identifies the QE save directory
  NAMELIST /inputpp/ prefix, outdir

  ! -------------------------------------------------------------------------
  ! Initialise MPI and the QE runtime environment
  ! -------------------------------------------------------------------------
  CALL mp_startup()
  CALL environment_start('PRINT_RHO_G')

  ! Guard: G-vectors are distributed across MPI tasks; printing requires all
  ! of them on a single process.  Abort early with a clear message rather than
  ! silently producing incomplete output.
  IF (nprocs > 1) THEN
    CALL errore('print_rho_g', &
      'This program must be run with a single MPI process (nprocs=1). ' // &
      'For parallel use, gather G-vectors via gvect%ig_l2g first.', 1)
  END IF

  ! -------------------------------------------------------------------------
  ! Read input namelist from stdin (defaults fall back to QE conventions)
  ! -------------------------------------------------------------------------
  prefix = 'pwscf'
  outdir = './'
  IF (ionode) THEN
    READ(5, inputpp, IOSTAT=ios)
    ! ios < 0: end-of-file before namelist -- use default values
    ! ios > 0: parse error in namelist
    IF (ios > 0) CALL errore('print_rho_g', &
      'Error reading &inputpp namelist from stdin', ios)
  END IF
  CALL mp_bcast(prefix, ionode_id, world_comm)
  CALL mp_bcast(outdir, ionode_id, world_comm)
  ! tmp_dir must be consistent with outdir for QE's internal file routines
  tmp_dir = TRIM(outdir) // '/'

  ! -------------------------------------------------------------------------
  ! Read the QE save directory.
  !
  ! read_file() is defined in PW/src/read_file.f90.  It reads:
  !   data-file-schema.xml  -> initialises cell_base, gvect, dfftp, lsda_mod
  !   charge-density.dat    -> populates rho%of_r (real-space grid, local part)
  !
  ! After this call all module variables listed in the USE statements above
  ! (tpiba, ngm, g, mill, dfftp, rho, nspin) are ready to use.
  !
  ! Note: in QE 6.x read_file() is a bare external subroutine.
  !       In QE >= 7.2 it may be wrapped in read_file_module; add the
  !       appropriate USE statement if the compiler complains.
  ! -------------------------------------------------------------------------
  CALL read_file()

  ! -------------------------------------------------------------------------
  ! Transform rho from real space to reciprocal space and print rho(G)
  ! -------------------------------------------------------------------------
  ALLOCATE(psic(dfftp%nnr))

  DO ispin = 1, nspin

    ! Copy real-space density into a complex work array (imaginary part = 0)
    psic = CMPLX(rho%of_r(:, ispin), 0.0_DP, KIND=DP)

    ! Forward FFT: psic(r) -> psic(G)  [in-place, on dfftp grid]
    ! Afterwards, psic(dfftp%nl(ig)) holds the Fourier coefficient for
    ! the ig-th G-vector (see "Normalisation note" in the file header).
    CALL fwfft('Rho', psic, dfftp)

    ! Print results to stdout (serial output from the ionode)
    IF (ionode) THEN
      WRITE(stdout, '(A,I2)') '# spin component:', ispin
      WRITE(stdout, '(A)') &
        '#   m1   m2   m3     Gx(1/Bohr)    Gy(1/Bohr)    Gz(1/Bohr)' // &
        '        Re(rho_G)           Im(rho_G)'

      ! g(:,ig)    -- G-vectors in units of tpiba; multiply to get 1/Bohr
      ! mill(:,ig) -- integer Miller indices (h, k, l)
      ! dfftp%nl(ig) -- linearised 3D FFT-grid index for G-vector ig
      DO ig = 1, ngm
        WRITE(stdout, '(3I5, 3F14.8, 2ES22.12)') &
          mill(1,ig), mill(2,ig), mill(3,ig), &
          g(1,ig)*tpiba, g(2,ig)*tpiba, g(3,ig)*tpiba, &
          REAL(psic(dfftp%nl(ig)), DP), AIMAG(psic(dfftp%nl(ig)))
      END DO
    END IF

  END DO

  DEALLOCATE(psic)

  ! -------------------------------------------------------------------------
  ! Clean up QE runtime
  ! -------------------------------------------------------------------------
  CALL environment_end('PRINT_RHO_G')

END PROGRAM print_rho_g
