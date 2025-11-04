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
  Spinner,
  Badge,
  Divider,
  useToast,
} from '@chakra-ui/react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

function App() {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const toast = useToast()

  const handleAnalyze = async () => {
    if (!input.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter text or a URL to analyze',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setLoading(true)
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        input_text: input.trim(),
      })

      setResult(response.data)
      toast({
        title: 'Analysis Complete',
        description: 'Your input has been analyzed successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    } catch (err) {
      let message = err.response?.data?.detail || 'Failed to analyze input'
      toast({
        title: 'Error',
        description: message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setLoading(false)
    }
  }

  const getTruthPercentage = () => (result ? Math.round(result.truth_probability * 100) : 0)
  const getProgressColor = () => {
    const p = getTruthPercentage()
    if (p >= 70) return 'green'
    if (p >= 40) return 'yellow'
    return 'red'
  }

  return (
    <Box minH="100vh" bg="gray.50" py={10}>
      <Container maxW="container.md">
        <VStack spacing={8} align="stretch">
          <Box textAlign="center">
            <Heading
              as="h1"
              size="2xl"
              bgGradient="linear(to-r, blue.400, purple.500)"
              bgClip="text"
            >
              Fake News Detector
            </Heading>
            <Text color="gray.600" fontSize="lg">
              Enter plain text or a URL — powered by RoBERTa AI
            </Text>
          </Box>

          <Box bg="white" borderRadius="xl" boxShadow="lg" p={8}>
            <VStack spacing={6} align="stretch">
              <Textarea
                placeholder="Paste text or enter a URL here..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                size="lg"
                minH="200px"
                isDisabled={loading}
                color="black"
              />

              <Button
                colorScheme="blue"
                onClick={handleAnalyze}
                isLoading={loading}
                loadingText="Analyzing..."
                size="lg"
                width="full"
              >
                Analyze
              </Button>

              {result && (
                <>
                  <Divider />
                  <VStack spacing={4} align="stretch">
                    <Heading as="h3" size="md" color="gray.700">
                      Result
                    </Heading>

                    <Badge
                      colorScheme={result.label === 'true' ? 'green' : 'red'}
                      fontSize="md"
                      px={3}
                      py={1}
                      borderRadius="full"
                      alignSelf="start"
                    >
                      {result.label === 'true' ? 'Likely True' : 'Likely False'}
                    </Badge>

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

                    <Alert status={result.label === 'true' ? 'success' : 'error'} borderRadius="md">
                      <AlertIcon />
                      <Box>
                        <AlertTitle>
                          {result.label === 'true'
                            ? 'This content appears truthful'
                            : 'This content may be unreliable'}
                        </AlertTitle>
                        <AlertDescription>
                          {result.label === 'true'
                            ? 'AI indicates the content is likely legitimate.'
                            : 'AI suggests this may contain misinformation.'}
                        </AlertDescription>
                      </Box>
                    </Alert>
                  </VStack>
                </>
              )}

              {loading && (
                <Box textAlign="center" py={8}>
                  <Spinner size="xl" color="blue.500" thickness="4px" />
                  <Text mt={4} color="gray.600">
                    Analyzing your input with AI...
                  </Text>
                </Box>
              )}
            </VStack>
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}

export default App
