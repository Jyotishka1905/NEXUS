import { Points, PointMaterial } from "@react-three/drei";
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function StarField() {
  const pointsRef = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const array = new Float32Array(900);

    for (let i = 0; i < 900; i += 3) {
      array[i] = (Math.random() - 0.5) * 25;
      array[i + 1] = (Math.random() - 0.5) * 18;
      array[i + 2] = (Math.random() - 0.5) * 20 - 4;
    }

    return array;
  }, []);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.008;
      pointsRef.current.rotation.x += delta * 0.002;
    }
  });

  return (
    <Points
      ref={pointsRef}
      positions={positions}
      stride={3}
    >
      <PointMaterial
        transparent
        color="#ffffff"
        size={0.025}
        sizeAttenuation
        depthWrite={false}
      />
    </Points>
  );
}