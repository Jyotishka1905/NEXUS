import { Canvas, useFrame } from "@react-three/fiber";
import {
  Environment,
  Stars,
  useTexture,
} from "@react-three/drei";
import { Suspense, useRef } from "react";
import * as THREE from "three";

/* =========================================================
   EARTH
========================================================= */

function Earth() {
  const earthRef = useRef<THREE.Group>(null);

  /* ---------------------------------------------------------
     LOCAL EARTH TEXTURES
     
     Files required:
     public/textures/earth.jpg
     public/textures/earth_normal.jpg
     public/textures/earth_specular.jpg
  --------------------------------------------------------- */

  const [
    earthTexture,
    normalTexture,
    specularTexture,
  ] = useTexture([
    "/textures/earth.jpg",
    "/textures/earth_normal.jpg",
    "/textures/earth_specular.jpg",
  ]);

  /* Earth texture is a color texture */
  earthTexture.colorSpace = THREE.SRGBColorSpace;

  /* ---------------------------------------------------------
     EARTH ROTATION
  --------------------------------------------------------- */

  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.055;
    }
  });

  return (
    <group ref={earthRef}>

      {/* =====================================================
          MAIN EARTH
      ===================================================== */}

      <mesh>
        <sphereGeometry args={[2.35, 96, 96]} />

        <meshPhongMaterial
          map={earthTexture}
          normalMap={normalTexture}
          specularMap={specularTexture}
          specular={new THREE.Color("#444444")}
          shininess={18}
        />
      </mesh>

      {/* =====================================================
          INNER ATMOSPHERE
      ===================================================== */}

      <mesh scale={1.025}>
        <sphereGeometry args={[2.35, 96, 96]} />

        <meshBasicMaterial
          color="#198cff"
          transparent
          opacity={0.10}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

      {/* =====================================================
          OUTER ATMOSPHERE
      ===================================================== */}

      <mesh scale={1.065}>
        <sphereGeometry args={[2.35, 64, 64]} />

        <meshBasicMaterial
          color="#1688ff"
          transparent
          opacity={0.055}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>

    </group>
  );
}


/* =========================================================
   SATELLITE MODEL
========================================================= */

function SatelliteModel() {
  return (
    <group scale={0.55}>

      {/* =====================================================
          MAIN BODY
      ===================================================== */}

      <mesh>
        <boxGeometry args={[0.42, 0.55, 0.42]} />

        <meshStandardMaterial
          color="#bfc9d6"
          metalness={0.85}
          roughness={0.25}
        />
      </mesh>


      {/* =====================================================
          LEFT SOLAR PANEL
      ===================================================== */}

      <mesh position={[-0.72, 0, 0]}>
        <boxGeometry args={[0.8, 0.035, 0.42]} />

        <meshStandardMaterial
          color="#0755a8"
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>


      {/* =====================================================
          RIGHT SOLAR PANEL
      ===================================================== */}

      <mesh position={[0.72, 0, 0]}>
        <boxGeometry args={[0.8, 0.035, 0.42]} />

        <meshStandardMaterial
          color="#0755a8"
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>


      {/* =====================================================
          SOLAR PANEL DETAIL
      ===================================================== */}

      <mesh position={[-0.72, 0.025, 0]}>
        <boxGeometry args={[0.78, 0.006, 0.38]} />

        <meshBasicMaterial color="#35aaff" />
      </mesh>

      <mesh position={[0.72, 0.025, 0]}>
        <boxGeometry args={[0.78, 0.006, 0.38]} />

        <meshBasicMaterial color="#35aaff" />
      </mesh>


      {/* =====================================================
          ANTENNA POLE
      ===================================================== */}

      <mesh position={[0, 0.5, 0]}>
        <cylinderGeometry
          args={[0.018, 0.018, 0.5, 12]}
        />

        <meshStandardMaterial
          color="#dce5ef"
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>


      {/* =====================================================
          ANTENNA
      ===================================================== */}

      <mesh position={[0, 0.77, 0]}>
        <sphereGeometry args={[0.09, 20, 20]} />

        <meshStandardMaterial
          color="#e8edf2"
          metalness={0.8}
          roughness={0.15}
        />
      </mesh>


      {/* =====================================================
          BLUE SCANNING LIGHT
      ===================================================== */}

      <mesh position={[0, -0.32, 0]}>
        <sphereGeometry args={[0.07, 16, 16]} />

        <meshBasicMaterial color="#00aaff" />
      </mesh>

    </group>
  );
}


/* =========================================================
   SATELLITE ORBIT
========================================================= */

function Satellite() {
  const satelliteRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!satelliteRef.current) return;

    const time = clock.getElapsedTime();

    /* -------------------------------------------------------
       ORBIT SETTINGS
    ------------------------------------------------------- */

    const radius = 4.1;
    const speed = 0.20;

    const angle = time * speed;


    /* -------------------------------------------------------
       CIRCULAR ORBIT
    ------------------------------------------------------- */

    satelliteRef.current.position.x =
      Math.cos(angle) * radius;

    satelliteRef.current.position.z =
      Math.sin(angle) * radius;


    /* -------------------------------------------------------
       SMALL VERTICAL MOVEMENT
       
       Makes the orbit feel more organic.
    ------------------------------------------------------- */

    satelliteRef.current.position.y =
      Math.sin(angle * 1.5) * 0.35;


    /* -------------------------------------------------------
       SATELLITE FOLLOWS ORBIT DIRECTION
    ------------------------------------------------------- */

    satelliteRef.current.rotation.y =
      -angle + Math.PI / 2;


    /* -------------------------------------------------------
       SMALL FLOATING ROTATION
    ------------------------------------------------------- */

    satelliteRef.current.rotation.z =
      Math.sin(time * 1.5) * 0.08;
  });

  return (
    <group ref={satelliteRef}>
      <SatelliteModel />
    </group>
  );
}


/* =========================================================
   ORBIT RINGS
========================================================= */

function OrbitRings() {
  const ringOne = useRef<THREE.Mesh>(null);
  const ringTwo = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {

    /* -------------------------------------------------------
       FIRST RING
    ------------------------------------------------------- */

    if (ringOne.current) {
      ringOne.current.rotation.z +=
        delta * 0.035;

      ringOne.current.rotation.y +=
        delta * 0.015;
    }


    /* -------------------------------------------------------
       SECOND RING
    ------------------------------------------------------- */

    if (ringTwo.current) {
      ringTwo.current.rotation.z -=
        delta * 0.025;

      ringTwo.current.rotation.x +=
        delta * 0.012;
    }
  });

  return (
    <group>

      {/* =====================================================
          MAIN ORBITAL PATH
      ===================================================== */}

      <mesh
        ref={ringOne}
        rotation={[
          Math.PI / 2.5,
          0,
          0,
        ]}
      >
        <torusGeometry
          args={[
            3.15,
            0.009,
            8,
            256,
          ]}
        />

        <meshBasicMaterial
          color="#1688ff"
          transparent
          opacity={0.65}
        />
      </mesh>


      {/* =====================================================
          SECONDARY ORBITAL PATH
      ===================================================== */}

      <mesh
        ref={ringTwo}
        rotation={[
          Math.PI / 2.1,
          0.5,
          0,
        ]}
      >
        <torusGeometry
          args={[
            3.55,
            0.006,
            8,
            256,
          ]}
        />

        <meshBasicMaterial
          color="#4abaff"
          transparent
          opacity={0.32}
        />
      </mesh>

    </group>
  );
}


/* =========================================================
   ORBIT PARTICLES
========================================================= */

function OrbitParticles() {
  const particlesRef = useRef<THREE.Points>(null);

  const particleCount = 80;

  const positions = new Float32Array(
    particleCount * 3
  );


  /* ---------------------------------------------------------
     CREATE PARTICLE POSITIONS
  --------------------------------------------------------- */

  for (let i = 0; i < particleCount; i++) {

    const angle =
      (i / particleCount) *
      Math.PI *
      2;

    const radius = 3.15;


    positions[i * 3] =
      Math.cos(angle) * radius;

    positions[i * 3 + 1] =
      (Math.random() - 0.5) * 0.25;

    positions[i * 3 + 2] =
      Math.sin(angle) * radius;
  }


  /* ---------------------------------------------------------
     PARTICLE ROTATION
  --------------------------------------------------------- */

  useFrame((_, delta) => {

    if (particlesRef.current) {
      particlesRef.current.rotation.y +=
        delta * 0.08;
    }

  });


  return (
    <points ref={particlesRef}>

      <bufferGeometry>

        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />

      </bufferGeometry>


      <pointsMaterial
        color="#4abaff"
        size={0.035}
        transparent
        opacity={0.9}
        sizeAttenuation
        depthWrite={false}
      />

    </points>
  );
}


/* =========================================================
   CAMERA MOVEMENT
========================================================= */

function CameraMotion() {

  useFrame(({ camera, clock }) => {

    const time =
      clock.getElapsedTime();


    /* -------------------------------------------------------
       HORIZONTAL CAMERA MOTION
    ------------------------------------------------------- */

    camera.position.x =
      Math.sin(time * 0.12) *
      0.18;


    /* -------------------------------------------------------
       VERTICAL CAMERA MOTION
    ------------------------------------------------------- */

    camera.position.y =
      0.2 +
      Math.sin(time * 0.1) *
      0.08;


    /* -------------------------------------------------------
       CAMERA TARGET
    ------------------------------------------------------- */

    camera.lookAt(0, 0, 0);

  });

  return null;
}


/* =========================================================
   MAIN SCENE CONTENT
========================================================= */

function SceneContent() {

  return (
    <>

      {/* =====================================================
          LIGHTING
      ===================================================== */}

      <ambientLight
        intensity={0.28}
      />


      <directionalLight
        position={[5, 4, 5]}
        intensity={3}
      />


      <directionalLight
        position={[-4, 1, -3]}
        intensity={0.6}
        color="#1688ff"
      />


      <pointLight
        position={[4, 2, 4]}
        intensity={1.5}
        color="#5abfff"
      />


      {/* =====================================================
          STAR FIELD
      ===================================================== */}

      <Stars
        radius={35}
        depth={20}
        count={1800}
        factor={2}
        saturation={0}
        fade
        speed={0.25}
      />


      {/* =====================================================
          EARTH
      ===================================================== */}

      <Earth />


      {/* =====================================================
          ORBIT RINGS
      ===================================================== */}

      <OrbitRings />

      <OrbitParticles />


      {/* =====================================================
          SATELLITE
      ===================================================== */}

      <Satellite />


      {/* =====================================================
          CAMERA
      ===================================================== */}

      <CameraMotion />


      {/* =====================================================
          ENVIRONMENT
      ===================================================== */}

      <Environment
        preset="night"
      />

    </>
  );
}


/* =========================================================
   NEXUS 3D SCENE
========================================================= */

export default function NexusScene() {

  return (
    <Canvas
      camera={{
        position: [0, 0.2, 8],
        fov: 42,
      }}

      dpr={[1, 1.5]}

      gl={{
        antialias: true,
        alpha: true,
      }}
    >

      <Suspense fallback={null}>

        <SceneContent />

      </Suspense>

    </Canvas>
  );
}