import { useState } from 'react'
import {
  Box,
  Container,
  Heading,
  Textarea,
  Button,
  VStack,
  HStack,
  Text,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useToast,
  Spinner,
  Badge,
  Divider,
  Link,
  Collapse,
} from '@chakra-ui/react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

// New component to render each piece of evidence
function EvidenceCard({ item }) {
  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} w="100%" bg="gray.50">
      <VStack align="stretch" spacing={2}>
        <Link href={item.url} isExternal fontWeight="bold" color="blue.600" noOfLines={1}>
          {item.source_title}
        </Link>
        <Text fontSize="sm" color="gray.700">
          "...{item.snippet}..."
        </Text>
      </VStack>
    </Box>
  )
}

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const toast = useToast()

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter some text to analyze',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text,
      })

      setResult(response.data)
      toast({
        title: 'Analysis Complete',
        description: 'Your text has been analyzed successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Failed to analyze text. Please try again.'
      setError(errorMessage)
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setLoading(false)
    }
  }

  const getTruthPercentage = () => {
    if (!result) return 0
    return Math.round(result.truth_probability * 100)
  }

  const getProgressColor = () => {
    const percentage = getTruthPercentage()
    if (percentage >= 70) return 'green'
    if (percentage >= 40) return 'yellow'
    return 'red'
  }

  const getResultLabel = () => {
    if (!result) return ''
    return result.label === 'true' ? 'Likely True' : 'Likely False'
  }

  const getResultBadgeColor = () => {
    if (!result) return 'gray'
    return result.label === 'true' ? 'green' : 'red'
  }

  return (
    <Box minH="100vh" bg="gray.50" py={10}>
      <Container maxW="container.md">
        <VStack spacing={8} align="stretch">
          {/* Header */}
          <Box textAlign="center">
            <Heading
              as="h1"
              size="2xl"
              bgGradient="linear(to-r, blue.400, purple.500)"
              bgClip="text"
              mb={2}
            >
              AI News Detector
            </Heading>
            <Text color="gray.600" fontSize="lg">
              Analysis by RoBERTa, Evidence by Google
            </Text>
          </Box>

          {/* Main Card */}
          <Box bg="white" borderRadius="xl" boxShadow="lg" p={8}>
            <VStack spacing={6} align="stretch">
              {/* Text Input */}
              <Box>
                <Text mb={2} fontWeight="semibold" color="gray.700">
                  Enter text to analyze:
                </Text>
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste a news article, social media post, or any text you want to verify..."
                  size="lg"
                  minH="150px"
                  resize="vertical"
                  focusBorderColor="blue.400"
                  color="gray.800"
                  _placeholder={{ color: "gray.400" }}
                />
              </Box>

              {/* Analyze Button */}
              <Button
                colorScheme="blue"
                size="lg"
                onClick={handleAnalyze}
                isLoading={loading}
                loadingText="Analyzing..."
                disabled={!text.trim() || loading}
                _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
                transition="all 0.2s"
              >
                Analyze Text
              </Button>

              {/* Add this block to display the error */}
              {error && (
                <Alert status="error" borderRadius="md">
                  <AlertIcon />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Results Section (AI Prediction) */}
              <Collapse in={result !== null} animateOpacity>
                <Divider />
                <VStack spacing={4} align="stretch" py={4}>
                  <HStack justify="space-between" align="center">
                    <Heading as="h3" size="md" color="gray.700">
                      AI Analysis
                    </Heading>
                    <Badge
                      colorScheme={getResultBadgeColor()}
                      fontSize="md"
                      px={3}
                      py={1}
                      borderRadius="full"
                    >
                      {getResultLabel()}
                    </Badge>
                  </HStack>

                  {/* Truth Probability */}
                  <Box>
                    <HStack justify="space-between" mb={2}>
                      <Text fontWeight="medium" color="gray.600">
                        Truth Probability
                      </Text>
                      <Text fontSize="2xl" fontWeight="bold" color={`${getProgressColor()}.500`}>
                        {getTruthPercentage()}%
                      </Text>
                    </HStack>
                    <Progress
                      value={getTruthPercentage()}
                      size="lg"
                      colorScheme={getProgressColor()}
                      borderRadius="full"
                      hasStripe
                      isAnimated
                    />
                  </Box>

                  {/* Interpretation */}
                  <Alert
                    status={result?.label === 'true' ? 'success' : 'error'}
                    borderRadius="md"
                  >
                    <AlertIcon />
                    <Box>
                      <AlertTitle>
                        {result?.label === 'true' 
                          ? 'This text appears to be truthful' 
                          : 'This text may contain misinformation'}
                      </AlertTitle>
                      <AlertDescription>
                        {result?.label === 'true'
                          ? 'The AI model has high confidence this content is legitimate.'
                          : 'The AI model suggests this content may be unreliable or false.'}
                      </AlertDescription>
                    </Box>
                  </Alert>
                </VStack>
              </Collapse>

              {/* Evidence Section (Google Search) */}
              <Collapse in={result !== null} animateOpacity>
                <Divider />
                <VStack spacing={4} align="stretch" py={4}>
                  <Heading as="h3" size="md" color="gray.700">
                    Supporting Web Evidence
                  </Heading>
                  {result?.evidence.length > 0 ? (
                    result.evidence.map((item, index) => (
                      <EvidenceCard key={index} item={item} />
                    ))
                  ) : (
                    <Text color="gray.500" fontSize="sm">
                      No web results found (or Google Search API is not configured).
                    </Text>
                  )}
                </VStack>
              </Collapse>

              {/* Loading State */}
              {loading && (
                <Box textAlign="center" py={8}>
                  <Spinner size="xl" color="blue.500" thickness="4px" />
                  <Text mt={4} color="gray.600">
                    Analyzing with AI and searching Google...
                  </Text>
                </Box>
              )}
            </VStack>
          </Box>

          {/* Footer Info */}
          <Box textAlign="center" color="gray.500" fontSize="sm">
            <Text>
              This tool uses a RoBERTa-large-mnli model for an initial linguistic analysis.
            </Text>
            <Text>
              Results are AI-generated and should be verified with the evidence provided.
            </Text>
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}

export default App