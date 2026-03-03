interface MedicalCrossLogoProps {
  size?: number;
  particleCount?: number;
}

export default function MedicalCrossLogo({ size = 120, particleCount = 24 }: MedicalCrossLogoProps) {
  const particles = Array.from({ length: particleCount }, (_, i) => {
    const angle = (Math.PI * 2 * i) / particleCount;
    const distance = size * 0.4 + Math.random() * size * 0.15;
    return {
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance,
      size: 3 + Math.random() * 4,
      delay: Math.random() * 0.5,
    };
  });

  const colors = ['#4ea1ff', '#6bb3ff', '#3d8fff', '#5aa7ff', '#7dc4ff'];

  return (
    <div className="relative flex items-center justify-center" style={{ width: `${size}px`, height: `${size}px` }}>
      {/* Animated Particles */}
      {particles.map((particle, i) => {
        const color = colors[i % colors.length];
        const floatDuration = 2 + Math.random();
        const animationName = `float-${i}`;
        
        return (
          <div key={i}>
            <div
              className="absolute rounded-full"
              style={{
                width: `${particle.size}px`,
                height: `${particle.size}px`,
                left: '50%',
                top: '50%',
                backgroundColor: color,
                transform: `translate(-50%, -50%) translate(${particle.x}px, ${particle.y}px)`,
                animation: `particleFadeIn 0.8s ease-out ${particle.delay}s forwards, ${animationName} ${floatDuration}s ease-in-out ${0.8 + particle.delay}s infinite alternate`,
                boxShadow: `0 0 10px ${color}80`,
                opacity: 0,
              }}
            />
            <style>{`
              @keyframes ${animationName} {
                from {
                  transform: translate(-50%, -50%) translate(${particle.x}px, ${particle.y}px) translateY(0px);
                }
                to {
                  transform: translate(-50%, -50%) translate(${particle.x}px, ${particle.y}px) translateY(-4px);
                }
              }
            `}</style>
          </div>
        );
      })}
      
      {/* Medical Cross */}
      <div className="relative z-10 flex items-center justify-center">
        <svg width={size / 2} height={size / 2} viewBox="0 0 120 120" fill="none">
          <path
            d="M45 0H75V45H120V75H75V120H45V75H0V45H45V0Z"
            fill="#f1f4f9"
            style={{
              filter: 'drop-shadow(0 4px 20px rgba(78, 161, 255, 0.4))',
            }}
          />
        </svg>
      </div>

      <style>{`
        @keyframes particleFadeIn {
          from {
            opacity: 0;
            scale: 0.5;
          }
          to {
            opacity: 1;
            scale: 1;
          }
        }
      `}</style>
    </div>
  );
}
