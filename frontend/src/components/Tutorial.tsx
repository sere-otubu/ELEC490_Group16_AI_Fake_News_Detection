import { useState, useEffect } from 'react';
import { X, ChevronRight, ChevronLeft, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface TutorialStep {
  id: string;
  title: string;
  description: string;
  targetSelector: string;
  position: 'top' | 'bottom' | 'left' | 'right' | 'center';
}

interface TutorialProps {
  isActive: boolean;
  onComplete: () => void;
  onSkip: () => void;
}

const tutorialSteps: TutorialStep[] = [
  {
    id: 'header',
    title: 'Welcome to EvidenceMD',
    description: 'This is your medical claim verification dashboard. Let\'s take a quick tour of the key features.',
    targetSelector: 'header',
    position: 'bottom',
  },
  {
    id: 'new-session',
    title: 'New Session Button',
    description: 'Click here to start a fresh analysis session. Each session focuses on verifying one medical claim.',
    targetSelector: '[data-tutorial="new-session"]',
    position: 'bottom',
  },
  {
    id: 'conversation',
    title: 'Conversation Area',
    description: 'Your medical claim and the AI\'s analysis will appear here. You\'ll see detailed verdicts, reasoning, and confidence scores.',
    targetSelector: '[data-tutorial="conversation"]',
    position: 'right',
  },
  {
    id: 'suggestions',
    title: 'Example Claims',
    description: 'Click any of these example claims to quickly test the system with common medical misinformation.',
    targetSelector: '[data-tutorial="suggestions"]',
    position: 'top',
  },
  {
    id: 'input-area',
    title: 'Claim Input',
    description: 'Type or paste your medical claim here. You can also use voice input, attach URLs, or upload images.',
    targetSelector: '[data-tutorial="input"]',
    position: 'top',
  },
  {
    id: 'multimodal',
    title: 'Multi-Modal Input',
    description: 'Extract text from web pages, images, or use voice input to submit claims in different formats.',
    targetSelector: '[data-tutorial="multimodal"]',
    position: 'top',
  },
  {
    id: 'sources',
    title: 'Source Documents',
    description: 'After analysis, verified sources and evidence will appear here. Click to view full documents or visit web sources.',
    targetSelector: '[data-tutorial="sources"]',
    position: 'left',
  },
];

export default function Tutorial({ isActive, onComplete, onSkip }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });

  // Reset to first step when tutorial becomes active
  useEffect(() => {
    if (isActive) {
      setCurrentStep(0);
    }
  }, [isActive]);

  useEffect(() => {
    if (!isActive) return;

    const updateHighlight = () => {
      const step = tutorialSteps[currentStep];
      const element = document.querySelector(step.targetSelector);
      
      if (element) {
        const rect = element.getBoundingClientRect();
        setHighlightRect(rect);
        
        // Calculate tooltip position based on step position
        const tooltipWidth = 400;
        const tooltipHeight = 200;
        const padding = 20;
        
        let top = 0;
        let left = 0;
        
        switch (step.position) {
          case 'bottom':
            top = rect.bottom + padding;
            left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
            break;
          case 'top':
            top = rect.top - tooltipHeight - padding;
            left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
            break;
          case 'left':
            top = rect.top + (rect.height / 2) - (tooltipHeight / 2);
            left = rect.left - tooltipWidth - padding;
            break;
          case 'right':
            top = rect.top + (rect.height / 2) - (tooltipHeight / 2);
            left = rect.right + padding;
            break;
          case 'center':
            top = window.innerHeight / 2 - tooltipHeight / 2;
            left = window.innerWidth / 2 - tooltipWidth / 2;
            break;
        }
        
        // Keep tooltip within viewport
        top = Math.max(padding, Math.min(top, window.innerHeight - tooltipHeight - padding));
        left = Math.max(padding, Math.min(left, window.innerWidth - tooltipWidth - padding));
        
        setTooltipPosition({ top, left });
        
        // Scroll element into view
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    };

    updateHighlight();
    window.addEventListener('resize', updateHighlight);
    window.addEventListener('scroll', updateHighlight, true);
    
    return () => {
      window.removeEventListener('resize', updateHighlight);
      window.removeEventListener('scroll', updateHighlight, true);
    };
  }, [currentStep, isActive]);

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    onSkip();
  };

  if (!isActive) return null;

  const step = tutorialSteps[currentStep];

  return (
    <>
      {/* Dark overlay */}
      <div className="fixed inset-0 z-[100] bg-black/10 transition-opacity" />
      
      {/* Spotlight effect */}
      {highlightRect && (
        <div
          className="fixed z-[101] pointer-events-none"
          style={{
            top: highlightRect.top - 4,
            left: highlightRect.left - 4,
            width: highlightRect.width + 8,
            height: highlightRect.height + 8,
            boxShadow: '0 0 0 4px rgba(78, 161, 255, 0.9), 0 0 0 9999px rgba(0, 0, 0, 0.1)',
            borderRadius: '12px',
            transition: 'all 0.3s ease-out',
          }}
        />
      )}
      
      {/* Tooltip */}
      <div
        className="fixed z-[102] w-[400px] rounded-2xl border border-primary/30 bg-card shadow-2xl shadow-primary/20 transition-all duration-300"
        style={{
          top: `${tooltipPosition.top}px`,
          left: `${tooltipPosition.left}px`,
        }}
      >
        <div className="p-6">
          {/* Header */}
          <div className="mb-4 flex items-start justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15">
                <Sparkles className="h-4 w-4 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-foreground">{step.title}</h3>
                <p className="text-xs text-muted-foreground">
                  Step {currentStep + 1} of {tutorialSteps.length}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleSkip}
              className="h-8 w-8 rounded-lg hover:bg-destructive/10 hover:text-destructive"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Description */}
          <p className="mb-6 text-sm leading-relaxed text-foreground/90">
            {step.description}
          </p>
          
          {/* Progress bar */}
          <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300"
              style={{ width: `${((currentStep + 1) / tutorialSteps.length) * 100}%` }}
            />
          </div>
          
          {/* Navigation */}
          <div className="flex items-center justify-between gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevious}
              disabled={currentStep === 0}
              className="gap-1"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSkip}
              className="text-muted-foreground hover:text-foreground"
            >
              Skip Tutorial
            </Button>
            
            <Button
              size="sm"
              onClick={handleNext}
              className="gap-1 bg-primary hover:bg-primary/90"
            >
              {currentStep === tutorialSteps.length - 1 ? 'Finish' : 'Next'}
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
