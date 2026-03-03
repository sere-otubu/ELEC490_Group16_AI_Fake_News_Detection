import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { Play, BookOpen } from 'lucide-react';

interface WelcomeScreenProps {
  children: ReactNode;
  onTutorialStart?: () => void;
  onAnalyzeStart?: () => void;
}

export default function WelcomeScreen({ children, onTutorialStart, onAnalyzeStart }: WelcomeScreenProps) {
  const [showWelcome, setShowWelcome] = useState(false);
  const [isAnimatingOut, setIsAnimatingOut] = useState(false);

  const handleDismiss = () => {
    setIsAnimatingOut(true);
    setTimeout(() => {
      localStorage.setItem('hasVisited', 'true');
      setShowWelcome(false);
    }, 600);
  };

  const handleAnalyzeClick = () => {
    handleDismiss();
    setTimeout(() => {
      onAnalyzeStart?.();
    }, 700);
  };

  const handleTutorialClick = () => {
    handleDismiss();
    setTimeout(() => {
      onTutorialStart?.();
    }, 700);
  };

  useEffect(() => {
    const hasVisited = localStorage.getItem('hasVisited');
    if (!hasVisited) {
      setShowWelcome(true);
    }
  }, []);

  if (!showWelcome) {
    return <>{children}</>;
  }

  const particles = Array.from({ length: 24 }, (_, i) => {
    const angle = (i / 24) * 360;
    const radius = 80 + Math.random() * 20;
    const size = 4 + Math.random() * 8;
    const delay = Math.random() * 0.5;
    
    return {
      angle,
      radius,
      size,
      delay,
      x: Math.cos((angle * Math.PI) / 180) * radius,
      y: Math.sin((angle * Math.PI) / 180) * radius,
    };
  });

  return (
    <>
      <div 
        className={`fixed inset-0 z-50 flex items-center justify-center bg-background transition-opacity duration-600 ${
          isAnimatingOut ? 'opacity-0' : 'opacity-100'
        }`}
      >
        <div 
          className={`flex flex-col items-center gap-12 transition-all duration-800 ${
            isAnimatingOut ? 'opacity-0 scale-95' : 'opacity-100 scale-100'
          }`}
          style={{
            animation: isAnimatingOut ? 'none' : 'fadeInScale 0.8s ease-out forwards'
          }}
        >
          {/* Medical Cross with Particles */}
          <div className="relative flex items-center justify-center" style={{ width: '240px', height: '240px' }}>
            {/* Animated Particles */}
            {particles.map((particle, i) => {
              const colors = ['#4ea1ff', '#6bb3ff', '#3d8fff', '#5aa7ff', '#7dc4ff'];
              const color = colors[i % colors.length];
              return (
                <div
                  key={i}
                  className="absolute rounded-full"
                  style={{
                    width: `${particle.size}px`,
                    height: `${particle.size}px`,
                    left: '50%',
                    top: '50%',
                    backgroundColor: color,
                    transform: `translate(-50%, -50%) translate(${particle.x}px, ${particle.y}px)`,
                    animation: `particleFadeIn 0.8s ease-out ${particle.delay}s forwards, float ${2 + Math.random()}s ease-in-out ${0.8 + particle.delay}s infinite alternate`,
                    boxShadow: `0 0 10px ${color}80`,
                    opacity: 0,
                  }}
                />
              );
            })}
            
            {/* Medical Cross */}
            <div className="relative z-10 flex items-center justify-center">
              <svg width="120" height="120" viewBox="0 0 120 120" fill="none">
                <path
                  d="M45 0H75V45H120V75H75V120H45V75H0V45H45V0Z"
                  fill="#f1f4f9"
                />
              </svg>
            </div>
          </div>
          
          {/* Text */}
          <div className="flex flex-col items-center gap-6">
            <div className="flex flex-col items-center gap-2">
              <h1 className="text-6xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent font-display tracking-wide">
                EvidenceMD
              </h1>
              <p className="text-xl text-muted-foreground font-medium">
                Medical Claim Verifier
              </p>
            </div>
            
            {/* Buttons */}
            <div className="flex items-center gap-4">
              <button
                onClick={handleAnalyzeClick}
                className="group flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-all duration-200 hover:bg-primary/90 hover:scale-105 hover:shadow-lg hover:shadow-primary/30"
              >
                <Play className="h-5 w-5 transition-transform group-hover:translate-x-0.5" />
                Let's Analyze
              </button>
              
              <button
                onClick={handleTutorialClick}
                className="group flex items-center gap-2 rounded-lg border-2 border-primary/30 bg-card/50 px-6 py-3 font-semibold text-foreground transition-all duration-200 hover:border-primary/60 hover:bg-card hover:scale-105"
              >
                <BookOpen className="h-5 w-5 transition-transform group-hover:rotate-12" />
                Tutorial
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div className={`transition-opacity duration-300 ${showWelcome ? 'opacity-0' : 'opacity-100'}`}>
        {children}
      </div>

      <style>{`
        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.95) translateY(10px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }
        
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
        
        @keyframes float {
          from {
            transform: translate(-50%, -50%) translate(var(--x), var(--y)) translateY(0px);
          }
          to {
            transform: translate(-50%, -50%) translate(var(--x), var(--y)) translateY(-8px);
          }
        }
      `}</style>
    </>
  );
}
